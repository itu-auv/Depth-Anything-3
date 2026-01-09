# Depth Anything 3 - Agent Instructions

## Project Overview

Depth Anything 3 (DA3) predicts spatially consistent geometry from arbitrary visual inputs. Single plain transformer backbone with depth-ray representation.

## Environment

**Package Manager:** `uv` (NOT conda/micromamba)

```bash
# Install dependencies
uv pip install -e .

# With optional deps
uv pip install -e ".[app]"  # Gradio UI
uv pip install -e ".[gs]"   # Gaussian splatting
uv pip install -e ".[all]"  # Everything
```

## CLI Usage

Entry point: `da3` (or `python -m depth_anything_3`)

### Common Commands

```bash
# Auto-detect input type (image/images/video/colmap)
da3 auto <input_path> --export-format npz --export-dir ./output

# Single image
da3 image <image.jpg> --export-format npz

# Directory of images
da3 images <images_dir> --export-format npz

# Video
da3 video <video.mp4> --fps 1.0

# COLMAP directory
da3 colmap <colmap_dir>

# Start backend server
da3 backend --port 8008

# Launch Gradio UI
da3 gradio --port 7860
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir` | `da3-large` | Model: da3-giant, da3-large, da3-base, da3-small, da3mono-large |
| `--export-format` | `glb` | Output: glb, npz, depth, ply |
| `--export-dir` | `./da3_export` | Output directory |
| `--process-res` | 504 | Processing resolution |
| `--device` | `cuda` | Device |

## Project Structure

```
src/depth_anything_3/
├── cli.py              # Typer CLI entry point
├── api.py              # Python API
├── model/              # Model architecture
│   ├── da3.py          # Main DA3 model
│   ├── dpt.py          # DPT decoder
│   └── dinov2/         # DINOv2 backbone
├── services/           # Backend services
│   ├── backend.py      # FastAPI backend
│   └── inference_service.py
├── app/                # Gradio app
└── utils/              # Utilities
    └── export/         # Export formats (glb, npz, depth_vis)
```

## Models

| Model | Use Case |
|-------|----------|
| `da3-giant` | Best quality, multi-view |
| `da3-large` | Balanced |
| `da3mono-large` | Monocular depth only (fast) |
| `da3metric-large` | Metric depth |

## Export Formats

- `npz`: Raw depth arrays (for downstream processing)
- `glb`: 3D point cloud visualization
- `depth`: Colorized depth images
- `ply`: Point cloud

## Running Tests

```bash
uv run python -m pytest
```

## Notes

- Models auto-download from HuggingFace on first use
- Default resolution 504px, increase for better quality
- GPU with 8GB+ VRAM recommended
