#!/usr/bin/env python3
"""Depth Anything 3 - ZeroMQ Backend Server"""

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


class DepthAnythingZMQServer:
    def __init__(self, model_name="da3-large", device="cuda", port=5555):
        self.device = device

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        self.model = DepthAnything3.from_pretrained(
            f"depth-anything/{model_name.upper()}"
        )
        self.model.to(device=device).eval()

        logger.setLevel(logging.INFO)
        logger.info(f"Ready: model={model_name}, device={device}, port={port}")

    def process_request(self, request):
        try:
            command = request.get("command", "")

            if command == "ping":
                return {"status": "success", "message": "pong"}

            if command in ("inference", "inference_batch"):
                is_batch = command == "inference_batch"
                images = request["images"] if is_batch else [request["image"]]
                process_res = request.get("process_res", 504)

                intrinsics_list = request.get("intrinsics") if is_batch else None
                extrinsics_list = request.get("extrinsics") if is_batch else None
                intrinsics = (
                    np.stack(intrinsics_list).astype(np.float32)
                    if intrinsics_list
                    else None
                )
                extrinsics = (
                    np.stack(extrinsics_list).astype(np.float32)
                    if extrinsics_list
                    else None
                )

                with torch.no_grad():
                    prediction = self.model.inference(
                        images,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        export_dir=None,
                        export_format="mini_npz",
                        process_res=process_res,
                        show_cameras=False,
                    )

                depths = [d.astype(np.float32) for d in prediction.depth]
                pred_intr = getattr(prediction, "intrinsics", None)
                intrinsics_out = (
                    [i.astype(np.float32) for i in pred_intr] if pred_intr else None
                )

                if is_batch:
                    return {
                        "status": "success",
                        "depth": depths,
                        "intrinsics": intrinsics_out,
                    }
                return {
                    "status": "success",
                    "depth": depths[0],
                    "intrinsics": intrinsics_out[0] if intrinsics_out else None,
                }

            logger.warning(f"Unknown command: {command}")
            return {"status": "error", "error": f"Unknown command: {command}"}

        except Exception as e:
            error_msg = f"{e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

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
    parser = argparse.ArgumentParser(description="Depth Anything 3 ZeroMQ Server")
    parser.add_argument(
        "--model",
        type=str,
        default="da3-large",
        choices=[
            "da3-small",
            "da3-base",
            "da3-large",
            "da3-giant",
            "da3nested-giant-large",
            "da3mono-large",
        ],
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

    server = DepthAnythingZMQServer(
        model_name=args.model, device=args.device, port=args.port
    )
    server.run()


if __name__ == "__main__":
    main()
