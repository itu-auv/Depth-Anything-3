import glob
import os
import torch
import numpy as np
from PIL import Image
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
model = model.to(device=device)

images_dir = "/home/emin/code/catkin_ws/src/auv-software/auv_vision/slalom_test_images"
output_dir = os.path.join(images_dir, "depth_output")
os.makedirs(output_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
print(f"Found {len(images)} images")

for img_path in images:
    print(f"Processing: {os.path.basename(img_path)}")
    prediction = model.inference([img_path])
    
    depth = prediction.depth[0]  # [H, W]
    
    # Normalize to 0-255 for visualization
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_vis = (depth_norm * 255).astype(np.uint8)
    
    # Save
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    Image.fromarray(depth_vis).save(os.path.join(output_dir, f"{base_name}_depth.png"))
    np.save(os.path.join(output_dir, f"{base_name}_depth.npy"), depth)
    print(f"  Saved depth map: {base_name}_depth.png")

print(f"\nDone! Output saved to: {output_dir}")
