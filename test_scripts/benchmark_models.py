#!/usr/bin/env python3
"""
Benchmark script for Depth Anything 3 models.
Handles both HuggingFace format and locally downloaded models.
"""

import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.cfg import create_object, load_config
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import time
import matplotlib.cm as cm
import argparse
from safetensors.torch import load_file


HF_CACHE = os.path.expanduser('~/.cache/huggingface/hub')


def create_metric_visualization(depth, output_path, grid_size=5):
    """Create visualization with metric depth values shown on grid."""
    h, w = depth.shape
    
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    colormap = cm.get_cmap('viridis')
    colored = (colormap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
    
    img = Image.fromarray(colored)
    draw = ImageDraw.Draw(img)
    
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
    except:
        font = ImageFont.load_default()
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            
            cell_depth = depth[y1:y2, x1:x2].mean()
            
            draw.rectangle([x1, y1, x2, y2], outline='white', width=1)
            
            text = f'{cell_depth:.2f}m'
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            bbox = draw.textbbox((cx, cy), text, font=font, anchor='mm')
            draw.rectangle(bbox, fill='black')
            draw.text((cx, cy), text, fill='white', font=font, anchor='mm')
    
    img.save(output_path)


def is_hf_format(model_name):
    """Check if model is in HuggingFace format (has blobs dir)."""
    short = model_name.replace('/', '--')
    cache_dir = os.path.join(HF_CACHE, f'models--{short}')
    return os.path.exists(os.path.join(cache_dir, 'blobs'))


def load_model_local(model_name, device):
    """Load model from local wget-downloaded files."""
    short = model_name.replace('/', '--')
    cache_dir = os.path.join(HF_CACHE, f'models--{short}')
    
    config_path = os.path.join(cache_dir, 'config.json')
    model_path = os.path.join(cache_dir, 'model.safetensors')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load via DepthAnything3 using local path
    model = DepthAnything3.from_pretrained(cache_dir)
    return model.to(device=device)


def load_model(model_name, device):
    """Load model, handling both HF and local formats."""
    if is_hf_format(model_name):
        print(f"  [HF format] Loading from HuggingFace cache...")
        model = DepthAnything3.from_pretrained(model_name)
    else:
        print(f"  [Local format] Loading from local cache...")
        short = model_name.replace('/', '--')
        cache_dir = os.path.join(HF_CACHE, f'models--{short}')
        model = DepthAnything3.from_pretrained(cache_dir)
    
    return model.to(device=device)


def run_benchmark(input_dir, output_dir, models_to_test=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if models_to_test is None:
        models_to_test = [
            ('depth-anything/DA3-LARGE', 'da3-large'),
            ('depth-anything/DA3-GIANT', 'da3-giant'),
            ('depth-anything/DA3METRIC-LARGE', 'da3metric-large'),
            ('depth-anything/DA3MONO-LARGE', 'da3mono-large'),
            ('depth-anything/DA3NESTED-GIANT-LARGE', 'da3nested-giant-large'),
        ]
    
    images = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    if not images:
        images = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    
    print(f'Found {len(images)} images in {input_dir}')
    
    timing_results = {}
    
    for model_name, short_name in models_to_test:
        model_output_dir = os.path.join(output_dir, short_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        print(f'\n{"="*60}')
        print(f'Loading {model_name}')
        print(f'{"="*60}')
        
        load_start = time.time()
        try:
            model = load_model(model_name, device)
        except Exception as e:
            print(f'ERROR loading {model_name}: {e}')
            continue
        load_time = time.time() - load_start
        print(f'Model loaded in {load_time:.2f}s')
        
        inference_times = []
        
        for i, img_path in enumerate(images):
            filename = os.path.basename(img_path)
            name, _ = os.path.splitext(filename)
            
            print(f'[{i+1}/{len(images)}] {filename}...', end=' ')
            
            inf_start = time.time()
            prediction = model.inference([img_path])
            inf_time = time.time() - inf_start
            inference_times.append(inf_time)
            print(f'{inf_time:.3f}s')
            
            depth = prediction.depth[0]
            
            # Save grayscale depth
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_uint8)
            depth_img.save(os.path.join(model_output_dir, f'{name}_depth.png'))
            
            # For metric models, also save grid visualization
            if 'METRIC' in model_name.upper() or 'NESTED' in model_name.upper():
                metric_path = os.path.join(model_output_dir, f'{name}_metric_grid.png')
                create_metric_visualization(depth, metric_path, grid_size=5)
        
        avg_time = sum(inference_times) / len(inference_times)
        timing_results[short_name] = {
            'load': load_time,
            'avg_inference': avg_time,
            'total_inference': sum(inference_times),
            'times': inference_times
        }
        
        print(f'Average: {avg_time:.3f}s | Total: {sum(inference_times):.2f}s')
        
        del model
        torch.cuda.empty_cache()
    
    # Print summary
    print('\n' + '='*60)
    print('BENCHMARK RESULTS')
    print('='*60)
    print(f'Images tested: {len(images)}')
    print()
    
    for model, times in timing_results.items():
        print(f'{model}:')
        print(f'  Load time:      {times["load"]:.2f}s')
        print(f'  Avg inference:  {times["avg_inference"]:.3f}s')
        print(f'  Total:          {times["total_inference"]:.2f}s')
        print()
    
    print('='*60)
    return timing_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Depth Anything 3 models')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save depth outputs')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Specific models to test (e.g., DA3-LARGE DA3METRIC-LARGE)')
    
    args = parser.parse_args()
    
    models = None
    if args.models:
        models = [(f'depth-anything/{m}', m.lower()) for m in args.models]
    
    run_benchmark(args.input_dir, args.output_dir, models)
