# Depth Anything 3 ZMQ Server API

A ZeroMQ-based server for running Depth Anything 3 inference.

## Overview

The server provides a simple request/response API over ZeroMQ (REP/REQ pattern) for depth estimation using Depth Anything 3 models.

## Starting the Server

```bash
python zmq_server.py --model <model_name> --port <port> --device <device> --log-level <level>
```

### Arguments

| Argument | Type | Default | Options | Description |
|----------|------|---------|---------|-------------|
| `--model` | string | `da3mono-large` | `da3-small`, `da3-base`, `da3-large`, `da3-giant`, `da3nested-giant-large`, `da3mono-large`, `da3metric-large` | Model to load |
| `--device` | string | `cuda` | `cuda`, `cpu` | Device to run inference on |
| `--port` | int | `5555` | - | TCP port to bind |
| `--log-level` | string | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | Logging verbosity |

### Example

```bash
python zmq_server.py --model da3metric-large --port 5555 --device cuda
```

## API Commands

All requests are Python dictionaries sent via ZeroMQ. Responses are also dictionaries.

### 1. Ping

Health check command.

**Request:**
```python
{"command": "ping"}
```

**Response:**
```python
{"status": "success", "message": "pong"}
```

### 2. Inference

Run depth estimation on a single image.

**Request:**
```python
{
    "command": "inference",
    "image": image_data,
    "process_res": 504
}
```

**Parameters:**
- `image`: Input image as numpy array (H,W,3) or PIL Image or file path string
- `process_res`: Processing resolution (default: 504)

**Response:**
```python
{
    "status": "success",
    "depth": depth_map
}
```

**Returns:**
- `depth`: 2D numpy array (H,W) of depth values (relative depth, or raw output before metric scaling)

## Client Example

```python
import zmq
import numpy as np
from PIL import Image

# Connect to server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Load image
image = np.array(Image.open("image.jpg"))

# Run inference
socket.send_pyobj({
    "command": "inference",
    "image": image,
    "process_res": 504
})
response = socket.recv_pyobj()

# Get depth map
depth = response["depth"]  # (H,W) numpy array
print(f"Depth shape: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]")

socket.close()
context.term()
```

## Models

### Metric Depth Models

**`da3metric-large`**
- Predicts metric depth (requires client-side metric scaling)
- Formula: `depth_meters = (raw_output * focal_length) / 300.0`
- Best for applications requiring real-world scale

### Non-Metric Models

**`da3mono-large`**, **`da3-large`**, **`da3-base`**, **`da3-small`**, **`da3-giant`**
- Predict relative depth (arbitrary scale)
- Best for depth ordering, depth-aware processing

**`da3nested-giant-large`**
- Predicts metric depth in meters
- Combines two models for improved metric accuracy

## Metric Scaling for da3metric-large

For `da3metric-large` model, apply metric scaling on client side:

```python
# Scaled intrinsics for output resolution (e.g., 630x476)
H, W = depth.shape
K_scaled = intrinsics.copy()
K_scaled[0] *= W / original_W
K_scaled[1] *= H / original_H

focal = (K_scaled[0, 0] + K_scaled[1, 1]) / 2.0
depth_meters = depth * (focal / 300.0)
```

**Important:** Use intrinsics scaled to match the processed depth resolution for accurate metric conversion.

## Error Handling

All responses include a `status` field:
- `status: "success"` - Request completed successfully
- `status: "error"` - Request failed, `error` field contains details

**Example error response:**
```python
{
    "status": "error",
    "error": "Intrinsics must be a 3x3 numpy array"
}
```

## Point Cloud Projection

The model requires image dimensions divisible by 14 (ViT patch size). Your images may be resized (e.g., 640x480 → 630x476). **To project depth to 3D points correctly**, scale your intrinsics to match the output depth resolution:

```python
# Original intrinsics for 640x480
K_original = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# After receiving depth from server
H, W = depth.shape  # e.g., (476, 630)

# Scale intrinsics to match output resolution
K_scaled = K_original.copy()
K_scaled[0] *= W / 640.0
K_scaled[1] *= H / 480.0

# Project to 3D
uu, vv = np.meshgrid(np.arange(W), np.arange(H))
Z = depth
X = (uu - K_scaled[0, 2]) * Z / K_scaled[0, 0]
Y = (vv - K_scaled[1, 2]) * Z / K_scaled[1, 1]
points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
```

This ensures 3D points are in correct world/metric space regardless of image resizing.

## Notes

- The server processes one request at a time (synchronous REP/REQ pattern)
- Images are resized to `process_res` while preserving aspect ratio
- Output dimensions are rounded to nearest multiple of 14 for ViT compatibility
- For `da3metric-large`, apply metric scaling on client side using scaled intrinsics
- The server logs performance metrics (FPS, VRAM) every 2 seconds when using CUDA
