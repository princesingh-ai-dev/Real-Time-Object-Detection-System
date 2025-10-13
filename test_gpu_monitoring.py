#!/usr/bin/env python3
"""
Test GPU Monitoring Functionality
Verify that GPU stats are being collected properly
"""

import sys
import time

def test_gpu_monitoring():
    """Test GPU monitoring capabilities"""
    print("=" * 70)
    print("GPU MONITORING TEST")
    print("=" * 70)
    
    # Test 1: Check PyTorch CUDA
    print("\n[1/4] Testing PyTorch CUDA...")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: True")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            # Test memory
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"   Total Memory: {gpu_mem_total:.2f} GB")
            
            # Allocate some memory
            test_tensor = torch.randn(1000, 1000, device='cuda')
            gpu_mem_used = torch.cuda.memory_allocated(0) / (1024 ** 2)
            print(f"   Memory Used: {gpu_mem_used:.2f} MB")
            
            del test_tensor
            torch.cuda.empty_cache()
            
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False
    
    # Test 2: Check pynvml
    print("\n[2/4] Testing pynvml (NVML)...")
    try:
        import pynvml
        
        pynvml.nvmlInit()
        print("✅ NVML Initialized")
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # GPU Name
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"   GPU Name: {name}")
        
        # GPU Utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"   GPU Usage: {utilization.gpu}%")
        print(f"   Memory Usage: {utilization.memory}%")
        
        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            print(f"   Temperature: {temp}°C")
        except:
            print("   Temperature: Not available")
        
        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            print(f"   Power: {power:.1f} W")
        except:
            print("   Power: Not available")
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"   VRAM Total: {mem_info.total / (1024**3):.2f} GB")
        print(f"   VRAM Used: {mem_info.used / (1024**3):.2f} GB")
        print(f"   VRAM Free: {mem_info.free / (1024**3):.2f} GB")
        
    except ImportError:
        print("❌ pynvml not installed")
        print("   Install with: pip install nvidia-ml-py3")
        return False
    except Exception as e:
        print(f"❌ pynvml test failed: {e}")
        return False
    
    # Test 3: Real-time monitoring
    print("\n[3/4] Testing real-time monitoring (5 seconds)...")
    try:
        print("   Monitoring GPU stats...")
        
        for i in range(5):
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_str = f"{temp}°C"
            except:
                temp_str = "N/A"
            
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_str = f"{power:.1f}W"
            except:
                power_str = "N/A"
            
            print(f"   [{i+1}/5] GPU: {utilization.gpu:>3}% | "
                  f"Mem: {mem_info.used / (1024**3):.2f}/{mem_info.total / (1024**3):.2f}GB | "
                  f"Temp: {temp_str:>6} | Power: {power_str:>7}")
            
            time.sleep(1)
        
        print("✅ Real-time monitoring working")
        
    except Exception as e:
        print(f"❌ Real-time monitoring failed: {e}")
        return False
    
    # Test 4: Stress test
    print("\n[4/4] GPU stress test (loading model)...")
    try:
        from ultralytics import YOLO
        
        print("   Loading YOLOv8 model...")
        model = YOLO('yolov8s.pt')
        
        # Check GPU usage
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"   After model load:")
        print(f"   GPU Usage: {utilization.gpu}%")
        print(f"   Memory: {mem_info.used / (1024**3):.2f} GB")
        
        # Run inference
        print("   Running inference test...")
        import numpy as np
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        results = model(test_img, device='cuda:0', verbose=False)
        
        # Check GPU usage during inference
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"   During inference:")
        print(f"   GPU Usage: {utilization.gpu}%")
        print(f"   Memory: {mem_info.used / (1024**3):.2f} GB")
        
        print("✅ GPU stress test passed")
        
    except Exception as e:
        print(f"⚠️  GPU stress test: {e}")
        print("   (This is optional, basic monitoring still works)")
    
    # Cleanup
    try:
        pynvml.nvmlShutdown()
    except:
        pass
    
    print("\n" + "=" * 70)
    print("✅ ALL GPU MONITORING TESTS PASSED!")
    print("=" * 70)
    print("\nYour GPU monitoring is working correctly!")
    print("The GUI should now display real-time GPU stats.")
    
    return True

if __name__ == "__main__":
    print("\n🎮 GPU Monitoring Test for RTX 2050\n")
    
    success = test_gpu_monitoring()
    
    if success:
        print("\n✅ GPU monitoring is ready!")
        print("   Run: python run_detection.py --gui")
        print("   Check the 🎮 GPU Monitor panel for real-time stats")
    else:
        print("\n❌ GPU monitoring test failed")
        print("   Check the errors above for troubleshooting")
    
    sys.exit(0 if success else 1)
