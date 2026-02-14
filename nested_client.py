#!/usr/bin/env python3
"""Depth Anything 3 Nested ROS Node - Multi-view ZMQ Client with pose refinement.

Buffers recent frames with their camera extrinsics (from tf), sends batched
inference requests to the DA3 nested ZMQ server, and publishes:
  - raw_depth        (sensor_msgs/Image, 32FC1, metric depth in meters)
  - colorized        (sensor_msgs/Image, bgr8, colorized depth visualization)
  - scaled_camera_info (sensor_msgs/CameraInfo, intrinsics at depth resolution)
  - refined_pose     (geometry_msgs/PoseWithCovarianceStamped, for robot_localization EKF)

The nested model (DA3NESTED-GIANT-LARGE-1.1) outputs metric depth directly in
meters (no focal/300 scaling needed) and returns refined camera poses aligned to
the input coordinate system via Umeyama Sim3.
"""

import collections

import cv2
import numpy as np
import rospy
import tf2_ros
import zmq
from auv_common_lib.vision.camera_calibrations import CameraCalibrationFetcher
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf.transformations
from sensor_msgs.msg import CameraInfo, Image

ZMQ_TIMEOUT_MS = 30000


class DepthAnythingNestedNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()

        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5556)
        self.rate_hz = rospy.get_param("~rate", 2.0)
        self.process_res = rospy.get_param("~process_res", 504)
        self.batch_size = rospy.get_param("~batch_size", 5)
        self.camera_namespace = rospy.get_param(
            "~camera_namespace", "cameras/cam_front"
        )
        self.frame_id = rospy.get_param(
            "~frame_id", "taluy/base_link/front_camera_optical_link"
        )
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.pose_covariance_diag = rospy.get_param(
            "~pose_covariance_diag",
            [0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
        )
        self.min_frame_interval = rospy.get_param("~min_frame_interval", 0.5)

        camera_info_fetcher = CameraCalibrationFetcher(
            self.camera_namespace, wait_for_camera_info=True
        )
        self.camera_info = camera_info_fetcher.get_camera_info()
        self.scaled_intrinsics = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.context = zmq.Context()
        self.socket = None
        self._reset_zmq()
        self.intrinsics_sent = False

        # Frame buffer: deque of (rgb, w2c_4x4, header)
        self.frame_buffer: collections.deque = collections.deque(maxlen=self.batch_size)
        self._last_buffer_time = 0.0

        self.latest_image = None
        self.image_sub = rospy.Subscriber(
            "image_raw", Image, self._image_cb, queue_size=1
        )

        self.depth_pub = rospy.Publisher("raw_depth", Image, queue_size=1)
        self.colorized_pub = rospy.Publisher("colorized", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher(
            "scaled_camera_info", CameraInfo, queue_size=1, latch=True
        )
        self.pose_pub = rospy.Publisher(
            "refined_pose", PoseWithCovarianceStamped, queue_size=1
        )

        rospy.on_shutdown(self.cleanup)

        if not self._ping_server():
            rospy.logwarn("[DA3] Server not responding. Will retry in loop.")
        rospy.loginfo(
            f"[DA3 Nested] Ready: {self.zmq_host}:{self.zmq_port}, "
            f"batch_size={self.batch_size}"
        )

    def _reset_zmq(self) -> None:
        if self.socket:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")
        self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.intrinsics_sent = False

    def _ping_server(self) -> bool:
        try:
            self.socket.send_pyobj({"command": "ping"})
            return self.socket.recv_pyobj().get("status") == "success"
        except zmq.error.Again:
            return False

    def _send_intrinsics(self) -> bool:
        """Send the camera intrinsics to the server (once)."""
        if self.camera_info is None:
            return False
        K = np.array(self.camera_info.K, dtype=np.float32).reshape(3, 3)
        try:
            self.socket.send_pyobj({"command": "set_intrinsics", "intrinsics": K})
            resp = self.socket.recv_pyobj()
            if resp.get("status") == "success":
                self.intrinsics_sent = True
                rospy.loginfo(
                    f"[DA3] Intrinsics sent: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}"
                )
                return True
            rospy.logerr(f"[DA3] set_intrinsics failed: {resp.get('error')}")
            return False
        except zmq.error.Again:
            rospy.logwarn("[DA3] Timeout sending intrinsics")
            return False

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = msg

    def _get_w2c(self, stamp: rospy.Time) -> np.ndarray | None:
        """Look up C2W from tf and invert to W2C (4x4) for DA3."""
        try:
            trans = self.tf_buffer.lookup_transform(
                self.world_frame, self.frame_id, stamp, rospy.Duration(0.1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(5.0, f"[DA3] TF lookup failed: {e}")
            return None

        t = trans.transform.translation
        q = trans.transform.rotation

        # quaternion_matrix expects [x, y, z, w]
        c2w = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w]).astype(
            np.float32
        )
        c2w[:3, 3] = [t.x, t.y, t.z]

        w2c = np.linalg.inv(c2w).astype(np.float32)
        return w2c

    def _w2c_to_pose_msg(
        self, w2c_3x4: np.ndarray, stamp: rospy.Time
    ) -> PoseWithCovarianceStamped:
        """Convert a (3,4) refined W2C matrix to PoseWithCovarianceStamped.

        Inverts W2C -> C2W to obtain the camera's pose in the world frame.
        """
        # Pad (3,4) -> (4,4) and invert to C2W
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = w2c_3x4.astype(np.float64)
        c2w = np.linalg.inv(w2c)

        x, y, z = c2w[:3, 3]
        qx, qy, qz, qw = tf.transformations.quaternion_from_matrix(c2w)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Diagonal covariance  [x, y, z, roll, pitch, yaw]
        cov = np.zeros(36, dtype=np.float64)
        for i in range(6):
            cov[i * 7] = self.pose_covariance_diag[i]
        msg.pose.covariance = cov.tolist()

        return msg

    @staticmethod
    def _colorize_depth(depth: np.ndarray) -> np.ndarray:
        d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        d_u8 = (d_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)

    def _update_intrinsics(self, h: int, w: int) -> None:
        """Recompute scaled intrinsics when depth output resolution changes."""
        if self.camera_info is None:
            return

        if (
            self.scaled_intrinsics is None
            or self.scaled_intrinsics["width"] != w
            or self.scaled_intrinsics["height"] != h
        ):
            orig_w = self.camera_info.width
            orig_h = self.camera_info.height
            if orig_w == 0 or orig_h == 0:
                rospy.logerr_throttle(
                    5.0, "[DA3] Camera resolution is 0, cannot scale intrinsics."
                )
                return

            scale_w = w / float(orig_w)
            scale_h = h / float(orig_h)

            self.scaled_intrinsics = {
                "width": w,
                "height": h,
                "fx": self.camera_info.K[0] * scale_w,
                "fy": self.camera_info.K[4] * scale_h,
                "cx": self.camera_info.K[2] * scale_w,
                "cy": self.camera_info.K[5] * scale_h,
            }
            rospy.loginfo(f"[DA3] Scaled intrinsics from {orig_w}x{orig_h} to {w}x{h}")
            self._publish_scaled_camera_info()

    def _publish_scaled_camera_info(self) -> None:
        if self.scaled_intrinsics is None or self.camera_info is None:
            return

        msg = CameraInfo()
        msg.header = self.camera_info.header
        msg.width = self.scaled_intrinsics["width"]
        msg.height = self.scaled_intrinsics["height"]
        msg.distortion_model = self.camera_info.distortion_model
        msg.D = self.camera_info.D

        fx = self.scaled_intrinsics["fx"]
        fy = self.scaled_intrinsics["fy"]
        cx = self.scaled_intrinsics["cx"]
        cy = self.scaled_intrinsics["cy"]

        msg.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.camera_info_pub.publish(msg)

    def _infer_batch(
        self,
        images: list[np.ndarray],
        extrinsics: np.ndarray,
    ) -> dict:
        """Send batched inference request. Returns server response dict."""
        self.socket.send_pyobj(
            {
                "command": "inference",
                "images": images,
                "extrinsics": extrinsics,
                "process_res": self.process_res,
            }
        )
        response = self.socket.recv_pyobj()
        if response["status"] != "success":
            raise RuntimeError(response.get("error", "Unknown error"))
        return response

    def run(self) -> None:
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue

            # Send intrinsics to server once
            if not self.intrinsics_sent:
                if not self._send_intrinsics():
                    rate.sleep()
                    continue

            # Grab latest image
            msg = self.latest_image
            self.latest_image = None

            try:
                # Add frame to sliding window buffer (throttled)
                now = msg.header.stamp.to_sec()
                if now - self._last_buffer_time >= self.min_frame_interval:
                    w2c = self._get_w2c(msg.header.stamp)
                    if w2c is not None:
                        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        self.frame_buffer.append((rgb, w2c, msg.header))
                        self._last_buffer_time = now

                if len(self.frame_buffer) == 0:
                    rate.sleep()
                    continue

                # Build batch arrays from buffer
                images = [f[0] for f in self.frame_buffer]
                extrinsics = np.stack(
                    [f[1] for f in self.frame_buffer], axis=0
                )  # (N, 4, 4)

                # Run batched inference
                result = self._infer_batch(images, extrinsics)

                # ── Publish depth (server returns latest frame only) ────
                depth = result["depth"]  # (H', W')

                h, w = depth.shape
                self._update_intrinsics(h, w)

                # Nested model outputs metric depth directly in meters
                depth_msg = self.bridge.cv2_to_imgmsg(depth, "32FC1")
                depth_msg.header = msg.header
                depth_msg.header.frame_id = self.frame_id
                self.depth_pub.publish(depth_msg)

                # ── Publish colorized depth ──────────────────────────────
                colorized = self._colorize_depth(depth)
                color_msg = self.bridge.cv2_to_imgmsg(colorized, "bgr8")
                color_msg.header = msg.header
                color_msg.header.frame_id = self.frame_id
                self.colorized_pub.publish(color_msg)

                # ── Publish refined pose (latest frame) ──────────────────
                refined_ext = result["extrinsics"][-1]  # (3, 4)
                pose_msg = self._w2c_to_pose_msg(refined_ext, msg.header.stamp)
                self.pose_pub.publish(pose_msg)

            except (zmq.error.ZMQError, zmq.error.Again) as e:
                rospy.logwarn_throttle(5.0, f"[DA3] ZMQ error: {e}. Reconnecting...")
                self._reset_zmq()
            except Exception as e:
                rospy.logerr_throttle(5.0, f"[DA3] Inference failed: {e}")

            rate.sleep()

    def cleanup(self) -> None:
        rospy.loginfo("[DA3] Shutting down...")
        self.socket.close(linger=0)
        self.context.term()


if __name__ == "__main__":
    rospy.init_node("depth_anything_nested_node")
    try:
        node = DepthAnythingNestedNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
