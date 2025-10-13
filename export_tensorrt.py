#!/usr/bin/env python3
"""
TensorRT Export Script for RTX 2050
Exports models/best.pt to TensorRT Engine with FP16 optimization
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def export_to_tensorrt():
    """Export YOLOv8 model to TensorRT Engine format"""
    
    print("=" * 80)
    print("TENSORRT EXPORT FOR RTX 2050")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available")
        print("   TensorRT requires GPU acceleration")
        return False
    
    print(f"\n✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch: {torch.__version__}")
    
    # Model path
    model_path = Path("models/best.pt")
    
    # If best.pt doesn't exist, use yolov8s.pt as demo
    if not model_path.exists():
        print(f"\n⚠️  Model not found: {model_path}")
        print("   Using yolov8s.pt for demonstration")
        print(f"   To use your custom model, place it at: {model_path.absolute()}")
        model_path = Path("yolov8s.pt")
        
        if not model_path.exists():
            print(f"❌ ERROR: No model found")
            return False
    
    print(f"\n📁 Model: {model_path}")
    
    # Export parameters
    params = {
        'format': 'engine',      # TensorRT Engine
        'imgsz': 640,           # Image size: 640x640
        'batch': 1,             # Batch size: 1 (real-time)
        'half': True,           # FP16 precision
        'device': 0,            # GPU 0 (RTX 2050)
        'workspace': 4,         # 4GB workspace
        'verbose': True,        # Show progress
        'simplify': True        # ONNX simplification
    }
    
    print("\n🔧 Export Parameters:")
    print(f"   • Format: TensorRT Engine (.engine)")
    print(f"   • Image Size: {params['imgsz']}x{params['imgsz']}")
    print(f"   • Batch Size: {params['batch']} (real-time streaming)")
    print(f"   • Precision: FP16 Half-Precision (2x speed boost)")
    print(f"   • Device: cuda:{params['device']} (RTX 2050)")
    print(f"   • Workspace: {params['workspace']} GB")
    
    try:
        # Load model
        print(f"\n[1/3] Loading PyTorch model...")
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully")
        
        # Export to TensorRT
        print(f"\n[2/3] Exporting to TensorRT Engine...")
        print("⏱️  This may take 5-10 minutes for optimization...")
        print("🔥 TensorRT is optimizing specifically for your RTX 2050...")
        
        engine_path = model.export(**params)
        
        print(f"\n✅ Export completed!")
        print(f"📁 TensorRT Engine: {engine_path}")
        
        # Verify export
        print(f"\n[3/3] Verifying export...")
        engine_file = Path(engine_path)
        
        if engine_file.exists():
            file_size_mb = engine_file.stat().st_size / (1024 * 1024)
            print(f"✅ Engine file size: {file_size_mb:.2f} MB")
            print("✅ Export verification: PASSED")
            
            print("\n" + "=" * 80)
            print("🎉 SUCCESS! TENSORRT ENGINE CREATED")
            print("=" * 80)
            print(f"\n📊 Output File: {engine_path}")
            print("\n🚀 Optimizations Applied:")
            print("   ✅ FP16 Half-Precision (2x speed on RTX 2050)")
            print("   ✅ GPU-specific kernel optimization")
            print("   ✅ Layer fusion for reduced overhead")
            print("   ✅ Optimized memory layout")
            print("   ✅ Dynamic tensor memory management")
            
            print("\n📈 Expected Performance:")
            print("   • 2-3x faster than PyTorch (.pt)")
            print("   • 80-150 FPS on RTX 2050 (model dependent)")
            print("   • 8-12 ms latency per frame")
            print("   • Real-time video processing ready")
            
            print("\n💡 Next Steps:")
            print("   1. Test performance: python test_trt_speed.py")
            print("   2. Use in application:")
            print(f"      model = YOLO('{engine_path}')")
            print("      results = model('image.jpg')")
            
            return engine_path
        else:
            print("❌ ERROR: Engine file not created")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR during export: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure CUDA PyTorch is installed")
        print("   2. Check GPU availability: nvidia-smi")
        print("   3. Verify model file exists")
        print("   4. Try with smaller workspace (workspace=2)")
        
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🚀 TensorRT Export Tool for RTX 2050")
    print("Exporting models/best.pt with optimal settings...\n")
    
    success = export_to_tensorrt()
    
    if success:
        print("\n✅ TensorRT optimization complete!")
        print("Your model is ready for maximum-speed inference on RTX 2050!\n")
        sys.exit(0)
    else:
        print("\n❌ Export failed. Check errors above.\n")
        sys.exit(1)
