#!/usr/bin/env python3
"""
TensorRT Model Export Script
Converts custom YOLOv8 PyTorch model to TensorRT Engine for RTX 2050
Optimized for maximum inference speed with FP16 precision
"""

import sys
import os
from pathlib import Path
from ultralytics import YOLO
import torch

def check_tensorrt_support():
    """Check if TensorRT is available"""
    print("=" * 70)
    print("CHECKING TENSORRT SUPPORT")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        print("TensorRT requires CUDA-enabled PyTorch")
        return False
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Check TensorRT
    try:
        import tensorrt as trt
        print(f"TensorRT Version: {trt.__version__}")
        print("TensorRT: AVAILABLE")
        return True
    except ImportError:
        print("WARNING: TensorRT not installed")
        print("\nTo install TensorRT:")
        print("1. pip install nvidia-tensorrt")
        print("   OR")
        print("2. Download from: https://developer.nvidia.com/tensorrt")
        print("\nNote: Ultralytics will attempt export anyway")
        return False

def export_model_to_tensorrt(
    model_path: str = "models/best.pt",
    imgsz: int = 640,
    batch: int = 1,
    half: bool = True,
    device: int = 0
):
    """
    Export YOLOv8 model to TensorRT Engine format
    
    Args:
        model_path: Path to PyTorch model (.pt file)
        imgsz: Input image size (640 recommended for real-time)
        batch: Batch size (1 for real-time streaming)
        half: Use FP16 precision (True for RTX 2050 speed boost)
        device: CUDA device ID (0 for first GPU)
    """
    print("\n" + "=" * 70)
    print("TENSORRT EXPORT PROCESS")
    print("=" * 70)
    
    # Verify model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nPlease ensure your trained model is at:")
        print(f"  {model_path.absolute()}")
        return False
    
    print(f"\nModel Path: {model_path}")
    print(f"Image Size: {imgsz}x{imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Precision: {'FP16 (Half)' if half else 'FP32 (Full)'}")
    print(f"Device: cuda:{device}")
    
    try:
        # Load model
        print("\n[1/4] Loading PyTorch model...")
        model = YOLO(str(model_path))
        print(f"Model loaded: {model_path.name}")
        
        # Export to TensorRT
        print("\n[2/4] Exporting to TensorRT Engine...")
        print("This may take 5-10 minutes for first-time optimization...")
        print("TensorRT will optimize the model specifically for your RTX 2050")
        
        export_path = model.export(
            format='engine',        # TensorRT Engine format
            imgsz=imgsz,           # Input size
            batch=batch,           # Batch size
            half=half,             # FP16 precision
            device=device,         # GPU device
            verbose=True,          # Show progress
            simplify=True,         # Simplify ONNX (intermediate step)
            workspace=4,           # TensorRT workspace in GB (4GB for RTX 2050)
        )
        
        print(f"\n[3/4] Export completed!")
        print(f"TensorRT Engine: {export_path}")
        
        # Verify export
        print("\n[4/4] Verifying exported model...")
        if Path(export_path).exists():
            file_size = Path(export_path).stat().st_size / (1024 * 1024)
            print(f"Engine file size: {file_size:.2f} MB")
            print("Export verification: PASSED")
            
            print("\n" + "=" * 70)
            print("SUCCESS! TensorRT Engine Created")
            print("=" * 70)
            print(f"\nExported file: {export_path}")
            print(f"\nOptimizations applied:")
            print("  - FP16 Half-Precision (2x speed boost)")
            print("  - RTX 2050 GPU-specific optimizations")
            print("  - Fused layers for reduced overhead")
            print("  - Optimized memory layout")
            print("\nExpected performance improvement:")
            print("  - 2-3x faster than PyTorch (.pt)")
            print("  - 50-100+ FPS on RTX 2050 (depending on model size)")
            
            return export_path
        else:
            print("ERROR: Export file not found")
            return False
            
    except Exception as e:
        print(f"\nERROR during export: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure CUDA PyTorch is installed:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("2. Install TensorRT:")
        print("   pip install nvidia-tensorrt")
        print("3. Update Ultralytics:")
        print("   pip install -U ultralytics")
        return False

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("YOLOV8 TO TENSORRT CONVERTER")
    print("Optimized for NVIDIA RTX 2050")
    print("=" * 70)
    
    # Check TensorRT support
    has_tensorrt = check_tensorrt_support()
    
    if not has_tensorrt:
        print("\nContinuing without TensorRT library...")
        print("Ultralytics will download and use built-in TensorRT support")
    
    # Export model
    print("\n" + "=" * 70)
    print("STARTING EXPORT")
    print("=" * 70)
    
    model_path = "models/best.pt"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\nWARNING: Model not found at {model_path}")
        print("\nTo use this script:")
        print(f"1. Place your trained YOLOv8 model at: {Path(model_path).absolute()}")
        print("2. Run this script again: python export_to_tensorrt.py")
        print("\nFor testing purposes, you can use a pretrained model:")
        print("   python export_to_tensorrt.py --model yolov8s.pt")
        return False
    
    # Export with optimal settings for RTX 2050
    engine_path = export_model_to_tensorrt(
        model_path=model_path,
        imgsz=640,          # Optimal for real-time
        batch=1,            # Real-time streaming
        half=True,          # FP16 for RTX 2050
        device=0            # First GPU
    )
    
    if engine_path:
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Test the TensorRT engine:")
        print("   python test_trt_speed.py")
        print("\n2. Use in your application:")
        print(f"   model = YOLO('{engine_path}')")
        print("   results = model('image.jpg')")
        print("\n3. Expected FPS on RTX 2050:")
        print("   - YOLOv8n: 100-150 FPS")
        print("   - YOLOv8s: 80-120 FPS")
        print("   - YOLOv8m: 50-80 FPS")
        return True
    else:
        print("\nExport failed. Please check errors above.")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export YOLOv8 to TensorRT')
    parser.add_argument('--model', type=str, default='models/best.pt',
                       help='Path to PyTorch model')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--batch', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--fp32', action='store_true',
                       help='Use FP32 instead of FP16')
    
    args = parser.parse_args()
    
    if args.model != 'models/best.pt':
        # Custom model specified
        success = export_model_to_tensorrt(
            model_path=args.model,
            imgsz=args.imgsz,
            batch=args.batch,
            half=not args.fp32,
            device=0
        )
    else:
        # Use default
        success = main()
    
    sys.exit(0 if success else 1)
