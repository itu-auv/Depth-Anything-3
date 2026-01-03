#!/usr/bin/env python3
"""
Depth Anything 3 - ZeroMQ Client
Client for sending images to the ZMQ inference server.
"""
import zmq
import numpy as np
import cv2
import argparse
from pathlib import Path


class DepthAnythingZMQClient:
    def __init__(self, host="localhost", port=5555, timeout=30000):
        """
        Initialize ZeroMQ client.
        
        Args:
            host: Server hostname
            port: Server port
            timeout: Request timeout in milliseconds
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to tcp://{host}:{port}")

    def ping(self):
        """Test server connectivity."""
        self.socket.send_pyobj({'command': 'ping'})
        response = self.socket.recv_pyobj()
        return response.get('message') == 'pong'

    def infer_single(self, image, process_res=504):
        """
        Run inference on a single image.
        
        Args:
            image: numpy array (H, W, 3) BGR or RGB, or path to image
            process_res: Processing resolution
            
        Returns:
            dict with 'depth' (H, W) and 'intrinsics' (3, 3)
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        request = {
            'command': 'inference',
            'image': image,
            'process_res': process_res
        }
        
        self.socket.send_pyobj(request)
        response = self.socket.recv_pyobj()
        
        if response['status'] != 'success':
            raise RuntimeError(f"Inference failed: {response.get('error')}")
        
        return {
            'depth': response['depth'],
            'intrinsics': response['intrinsics']
        }

    def infer_batch(self, images, intrinsics=None, extrinsics=None, process_res=504):
        """
        Run inference on a batch of images.
        
        Args:
            images: List of numpy arrays (H, W, 3) or paths
            intrinsics: Optional list of (3, 3) intrinsic matrices
            extrinsics: Optional list of (4, 4) extrinsic matrices
            process_res: Processing resolution
            
        Returns:
            dict with 'depth' list and 'intrinsics' list
        """
        processed_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = cv2.imread(str(img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_images.append(img)
        
        request = {
            'command': 'inference_batch',
            'images': processed_images,
            'process_res': process_res
        }
        
        if intrinsics is not None:
            request['intrinsics'] = intrinsics
        if extrinsics is not None:
            request['extrinsics'] = extrinsics
        
        self.socket.send_pyobj(request)
        response = self.socket.recv_pyobj()
        
        if response['status'] != 'success':
            raise RuntimeError(f"Inference failed: {response.get('error')}")
        
        return {
            'depth': response['depth'],
            'intrinsics': response['intrinsics']
        }

    def close(self):
        """Close the connection."""
        self.socket.close()
        self.context.term()


def visualize_depth(depth, colormap=cv2.COLORMAP_INFERNO, inverse=False):
    """Convert depth map to colorized visualization."""
    if inverse:
        # Visualize disparity (1/depth)
        depth = 1.0 / (depth + 1e-8)
        
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, colormap)


def main():
    parser = argparse.ArgumentParser(description='Depth Anything 3 ZMQ Client')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--process-res', type=int, default=504, help='Processing resolution')
    parser.add_argument('--output', type=str, default=None, help='Output path for depth visualization')
    parser.add_argument('--show', action='store_true', help='Display result')
    
    args = parser.parse_args()
    
    client = DepthAnythingZMQClient(host=args.host, port=args.port)
    
    # Test connection
    if not client.ping():
        print("Failed to connect to server")
        return
    print("Server connected")
    
    # Run inference
    print(f"Running inference on {args.image}...")
    result = client.infer_single(args.image, process_res=args.process_res)
    
    depth = result['depth']
    print(f"Depth shape: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]")
    
    # Visualize
    depth_vis = visualize_depth(depth)
    depth_vis_inv = visualize_depth(depth, inverse=True)
    
    # Save or show
    output_path = args.output or Path(args.image).stem + "_depth.png"
    output_path_inv = str(Path(output_path).with_stem(Path(output_path).stem + "_inv"))
    
    cv2.imwrite(output_path, depth_vis)
    cv2.imwrite(output_path_inv, depth_vis_inv)
    
    # Save raw
    np.save(str(Path(output_path).with_suffix('.npy')), depth)
    
    print(f"Saved depth visualization to {output_path}")
    print(f"Saved inverse depth visualization to {output_path_inv}")
    print(f"Saved raw depth to {Path(output_path).with_suffix('.npy')}")
    
    if args.show:
        cv2.imshow("Depth", depth_vis)
        cv2.imshow("Inverse Depth", depth_vis_inv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    client.close()


if __name__ == '__main__':
    main()
