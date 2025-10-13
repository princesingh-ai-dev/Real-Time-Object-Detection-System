#!/usr/bin/env python3
"""
TensorRT Speed Test Script
Verifies TensorRT engine performance on RTX 2050
"""

import sys
import time
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

def test_tensorrt_speed(engine_path: str = None, num_iterations: int = 100):
    """
    Test TensorRT engine inference speed
    
    Args:
        engine_path: Path to TensorRT engine file
        num_iterations: Number of test iterations
    """
    print("=" * 80)
    print("TENSORRT SPEED TEST - RTX 2050")
    print("=" * 80)
    
    # Find engine file
    if engine_path is None:
        # Look for engine files
        possible_engines = list(Path(".").glob("*.engine"))
        possible_engines.extend(list(Path("models").glob("*.engine")))
        
        if possible_engines:
            engine_path = str(possible_engines[0])
            print(f"✅ Found engine: {engine_path}")
        else:
            print("❌ No .engine file found")
            print("   Run: python export_tensorrt.py")
            return False
    
    engine_file = Path(engine_path)
    if not engine_file.exists():
        print(f"❌ Engine file not found: {engine_path}")
        return False
    
    # Check GPU
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available")
        print("   TensorRT may run on CPU (slower)")
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA: {torch.version.cuda}")
    
    print(f"\n📁 Engine: {engine_path}")
    print(f"📊 File Size: {engine_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"🔢 Test Iterations: {num_iterations}")
    
    try:
        # Load TensorRT model
        print("\n[1/4] Loading TensorRT engine...")
        model = YOLO(str(engine_path))
        print("✅ TensorRT engine loaded successfully")
        
        # Create placeholder image (640x640x3)
        print("\n[2/4] Creating test image (640x640x3)...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print("✅ Test image created")
        
        # Warmup phase
        print(f"\n[3/4] GPU warmup ({10} iterations)...")
        for i in range(10):
            _ = model(test_image, verbose=False)
        print("✅ Warmup complete")
        
        # Speed test
        print(f"\n[4/4] Running speed test ({num_iterations} iterations)...")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        for i in range(num_iterations):
            results = model(test_image, verbose=False)
            
            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                current_fps = (i + 1) / elapsed
                print(f"   Progress: {i + 1}/{num_iterations} - Current FPS: {current_fps:.1f}")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_iterations) * 1000
        fps = num_iterations / total_time
        
        print("\n" + "=" * 80)
        print("📊 TENSORRT PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"\n⏱️  Total Time: {total_time:.3f} seconds")
        print(f"⏱️  Average Inference Time: {avg_time_ms:.2f} ms/frame")
        print(f"🚀 Throughput: {fps:.1f} FPS")
        
        # Performance assessment
        print("\n" + "=" * 80)
        print("📈 PERFORMANCE ASSESSMENT")
        print("=" * 80)
        
        if fps >= 100:
            rating = "🌟 EXCELLENT! Ultra-fast real-time"
            use_case = "High-speed tracking, multi-camera systems"
        elif fps >= 60:
            rating = "✅ VERY GOOD! High-speed real-time"
            use_case = "Real-time detection, smooth video processing"
        elif fps >= 30:
            rating = "✅ GOOD! Real-time capable"
            use_case = "Video processing, standard applications"
        elif fps >= 15:
            rating = "⚠️  ACCEPTABLE: Near real-time"
            use_case = "Most video applications"
        else:
            rating = "❌ SLOW: Below real-time"
            use_case = "Consider smaller model or lower resolution"
        
        print(f"\n{rating}")
        print(f"Use Case: {use_case}")
        
        # GPU info
        if torch.cuda.is_available():
            print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"💾 GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Speedup estimate
        print("\n" + "=" * 80)
        print("🚀 EXPECTED SPEEDUP vs PyTorch")
        print("=" * 80)
        print("   • PyTorch (.pt): ~30-40 FPS")
        print(f"   • TensorRT (.engine): {fps:.1f} FPS")
        
        if fps > 40:
            speedup = fps / 35  # Assuming ~35 FPS for PyTorch
            print(f"   • Speedup: ~{speedup:.1f}x faster! 🎉")
        
        print("\n" + "=" * 80)
        print("✅ TENSORRT ENGINE VERIFIED AND READY!")
        print("=" * 80)
        print("\n💡 Use in your application:")
        print(f"   model = YOLO('{engine_path}')")
        print("   results = model('image.jpg')  # Fast inference!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_pytorch_vs_tensorrt():
    """Compare PyTorch and TensorRT performance"""
    print("\n" + "=" * 80)
    print("PYTORCH vs TENSORRT COMPARISON")
    print("=" * 80)
    
    # Find files
    pt_model = None
    engine_model = None
    
    for pt_path in ["models/best.pt", "yolov8s.pt"]:
        if Path(pt_path).exists():
            pt_model = pt_path
            break
    
    for eng_path in list(Path(".").glob("*.engine")) + list(Path("models").glob("*.engine")):
        engine_model = str(eng_path)
        break
    
    if not pt_model or not engine_model:
        print("⚠️  Comparison requires both .pt and .engine files")
        return
    
    print(f"\n📁 PyTorch: {pt_model}")
    print(f"📁 TensorRT: {engine_model}")
    
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = {}
    
    # Test PyTorch
    try:
        print("\n[1/2] Testing PyTorch model...")
        model_pt = YOLO(pt_model)
        
        for _ in range(5):
            _ = model_pt(test_img, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(50):
            _ = model_pt(test_img, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        fps_pt = 50 / elapsed
        results['pytorch'] = fps_pt
        print(f"✅ PyTorch: {fps_pt:.1f} FPS")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
    
    # Test TensorRT
    try:
        print("\n[2/2] Testing TensorRT engine...")
        model_trt = YOLO(engine_model)
        
        for _ in range(5):
            _ = model_trt(test_img, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(50):
            _ = model_trt(test_img, verbose=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        fps_trt = 50 / elapsed
        results['tensorrt'] = fps_trt
        print(f"✅ TensorRT: {fps_trt:.1f} FPS")
    except Exception as e:
        print(f"❌ TensorRT test failed: {e}")
    
    # Show comparison
    if 'pytorch' in results and 'tensorrt' in results:
        print("\n" + "=" * 80)
        print("📊 COMPARISON RESULTS")
        print("=" * 80)
        
        speedup = results['tensorrt'] / results['pytorch']
        improvement = (speedup - 1) * 100
        
        print(f"\n   PyTorch:   {results['pytorch']:>6.1f} FPS")
        print(f"   TensorRT:  {results['tensorrt']:>6.1f} FPS")
        print(f"\n   Speedup:   {speedup:.2f}x faster")
        print(f"   Improvement: {improvement:.1f}%")
        
        if speedup >= 2.5:
            print("\n   🌟 EXCELLENT! Major performance boost")
        elif speedup >= 1.5:
            print("\n   ✅ GOOD! Significant improvement")
        elif speedup >= 1.2:
            print("\n   ✅ MODERATE: Noticeable speedup")
        else:
            print("\n   ⚠️  MINIMAL: Limited improvement")

def main():
    """Main execution"""
    print("\n🚀 TensorRT Speed Test for RTX 2050\n")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available")
        print("   TensorRT works best with GPU acceleration\n")
    
    # Run speed test
    success = test_tensorrt_speed(num_iterations=100)
    
    if success:
        # Run comparison if PyTorch model available
        compare_pytorch_vs_tensorrt()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
