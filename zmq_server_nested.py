#!/usr/bin/env python3
"""Depth Anything 3 - Nested ZMQ Server for pose refinement.

Uses DA3NESTED-GIANT-LARGE-1.1 with align_to_input_ext_scale=False
to return the model's refined extrinsics (aligned to your coordinate system).

Protocol (pyobj over ZMQ REP):
    set_intrinsics:
        request:  {"command": "set_intrinsics", "intrinsics": np.ndarray (3,3) float32}
        response: {"status": "success"}

    inference:
        request:  {
            "command": "inference",
            "images": [np.ndarray (H,W,3) uint8, ...],        # N images (RGB)
            "extrinsics": np.ndarray (N,4,4) float32,          # W2C, OpenCV/COLMAP convention
            "process_res": int (optional, default 504),
        }
        response: {
            "status": "success",
            "depth": np.ndarray (H',W') float32,               # metric depth (latest frame only)
            "extrinsics": np.ndarray (N,3,4) float32,          # refined W2C poses (all frames)
            "intrinsics": np.ndarray (N,3,3) float32,          # (your input, echoed back)
            "conf": np.ndarray (H',W') float32,                # confidence map (latest frame only)
        }

    ping:
        request:  {"command": "ping"}
        response: {"status": "success", "message": "pong"}

Intrinsics must be set before calling inference.
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import zmq

sys.path.insert(0, str(Path(__file__).parent / "src"))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger as da3_logger

da3_logger.level = 1
logging.getLogger("dinov2").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class NestedDepthZMQServer:
    def __init__(
        self, model_name="DA3NESTED-GIANT-LARGE-1.1", device="cuda", port=5555
    ):
        self.device = device
        self.model_name = model_name
        self.intrinsics: np.ndarray | None = None  # (3, 3) float32

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        logger.info(f"Loading model depth-anything/{model_name} ...")
        self.model = DepthAnything3.from_pretrained(f"depth-anything/{model_name}")
        self.model.to(device=device).eval()

        logger.info(f"Ready: model={model_name}, device={device}, port={port}")

    def process_request(self, request):
        try:
            command = request.get("command", "")

            if command == "ping":
                return {"status": "success", "message": "pong"}

            if command == "set_intrinsics":
                K = request["intrinsics"]
                assert isinstance(K, np.ndarray) and K.shape == (3, 3), (
                    f"intrinsics must be (3,3) ndarray, got {type(K)} shape={getattr(K, 'shape', '?')}"
                )
                self.intrinsics = K.astype(np.float32)
                logger.info(
                    f"Intrinsics set: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} "
                    f"cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}"
                )
                return {"status": "success"}

            if command == "inference":
                return self._handle_inference(request)

            logger.warning(f"Unknown command: {command}")
            return {"status": "error", "error": f"Unknown command: {command}"}

        except Exception as e:
            error_msg = f"{e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

    def _handle_inference(self, request):
        if self.intrinsics is None:
            return {
                "status": "error",
                "error": "Intrinsics not set. Call set_intrinsics first.",
            }

        images = request["images"]  # list of (H, W, 3) uint8
        extrinsics = request["extrinsics"]  # (N, 4, 4) float32
        process_res = request.get("process_res", 504)

        N = len(images)
        assert extrinsics.shape == (N, 4, 4), (
            f"extrinsics shape mismatch: got {extrinsics.shape}, expected ({N}, 4, 4)"
        )

        # Broadcast single intrinsics to (N, 3, 3)
        intrinsics = np.tile(self.intrinsics, (N, 1, 1))

        with torch.no_grad():
            prediction = self.model.inference(
                images,
                extrinsics=extrinsics.astype(np.float32),
                intrinsics=intrinsics,
                align_to_input_ext_scale=False,
                process_res=process_res,
            )

        # Only return depth/conf for the latest frame to reduce payload size.
        # Extrinsics/intrinsics are still returned for all N frames (small).
        return {
            "status": "success",
            "depth": prediction.depth[-1].astype(np.float32),  # (H', W')
            "extrinsics": prediction.extrinsics.astype(np.float32),  # (N, 3, 4)
            "intrinsics": prediction.intrinsics.astype(np.float32),  # (N, 3, 3)
            "conf": prediction.conf[-1].astype(np.float32),  # (H', W')
        }

    def run(self):
        frame_count = 0
        last_log_time = time.time()

        try:
            while True:
                message = self.socket.recv_pyobj()
                response = self.process_request(message)
                self.socket.send_pyobj(response)

                frame_count += 1
                now = time.time()
                if now - last_log_time >= 2.0:
                    fps = frame_count / (now - last_log_time)
                    if self.device == "cuda":
                        alloc = torch.cuda.memory_allocated() / 1024**2
                        res = torch.cuda.memory_reserved() / 1024**2
                        logger.info(f"VRAM: {alloc:.0f}/{res:.0f} MB | Hz: {fps:.2f}")
                    else:
                        logger.info(f"Hz: {fps:.2f}")
                    frame_count = 0
                    last_log_time = now

        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            traceback.print_exc()
        finally:
            self.socket.close()
            self.context.term()
            logger.info("Server closed")


def main():
    parser = argparse.ArgumentParser(
        description="DA3 Nested ZMQ Server (pose refinement)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DA3NESTED-GIANT-LARGE-1.1",
        help="HuggingFace model name (without depth-anything/ prefix)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    server = NestedDepthZMQServer(
        model_name=args.model, device=args.device, port=args.port
    )
    server.run()


if __name__ == "__main__":
    main()
