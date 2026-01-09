#!/usr/bin/env python3
"""
Test script to verify all DA3 models are properly downloaded and loadable.
"""

import torch
import os
import sys

HF_CACHE = os.path.expanduser('~/.cache/huggingface/hub')

MODELS = [
    'depth-anything/DA3-BASE',
    'depth-anything/DA3-LARGE',
    'depth-anything/DA3-GIANT',
    'depth-anything/DA3METRIC-LARGE',
    'depth-anything/DA3MONO-LARGE',
    'depth-anything/DA3NESTED-GIANT-LARGE',
]


def check_cache():
    """Check which models exist in HuggingFace cache."""
    print("Checking HuggingFace cache...")
    print(f"Cache path: {HF_CACHE}\n")
    
    for model in MODELS:
        short = model.replace('/', '--')
        cache_dir = os.path.join(HF_CACHE, f'models--{short}')
        
        if os.path.exists(cache_dir):
            # Check size
            total_size = 0
            for dirpath, _, filenames in os.walk(cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            size_gb = total_size / (1024**3)
            
            # Check for required files
            has_blobs = os.path.exists(os.path.join(cache_dir, 'blobs'))
            has_config = os.path.exists(os.path.join(cache_dir, 'config.json'))
            has_model = os.path.exists(os.path.join(cache_dir, 'model.safetensors'))
            
            if has_blobs:
                status = "✅ HF format"
            elif has_config and has_model:
                status = "⚠️  wget format (config+model)"
            elif has_model:
                status = "❌ wget format (missing config)"
            else:
                status = "❓ unknown format"
            
            print(f"{model}: {status} ({size_gb:.2f} GB)")
        else:
            print(f"{model}: ❌ NOT DOWNLOADED")
    
    print()


def test_load_models():
    """Try to load each model."""
    from depth_anything_3.api import DepthAnything3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTesting model loading on {device}...")
    print()
    
    results = {}
    
    for model_name in MODELS:
        print(f"Loading {model_name}...", end=' ')
        try:
            model = DepthAnything3.from_pretrained(model_name)
            model = model.to(device=device)
            
            # Quick inference test
            import numpy as np
            from PIL import Image
            test_img = Image.new('RGB', (224, 224), color='blue')
            test_path = '/tmp/test_img.png'
            test_img.save(test_path)
            
            prediction = model.inference([test_path])
            depth_shape = prediction.depth.shape
            
            print(f"✅ OK (output shape: {depth_shape})")
            results[model_name] = True
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results[model_name] = False
    
    print()
    print("="*50)
    print("SUMMARY")
    print("="*50)
    passed = sum(1 for v in results.values() if v)
    print(f"Passed: {passed}/{len(MODELS)}")
    
    return all(results.values())


if __name__ == '__main__':
    check_cache()
    
    if '--load' in sys.argv:
        success = test_load_models()
        sys.exit(0 if success else 1)
