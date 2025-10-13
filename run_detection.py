#!/usr/bin/env python3
"""
Real-Time Object Detection System - Main Runner
Clean, simple interface for object detection
"""

import sys
import os
import argparse
from pathlib import Path

def check_requirements():
    """Check and install required packages"""
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'opencv-python', 'numpy', 'PyQt6', 'psutil'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing requirements...")
        try:
            os.system(f"{sys.executable} -m pip install -r requirements.txt")
            print("✅ Requirements installed successfully!")
        except Exception as e:
            print(f"❌ Failed to install requirements: {e}")
            print("Please install manually: pip install -r requirements.txt")
            return False

    return True

def run_gui_mode():
    """Run GUI mode"""
    try:
        print("🚀 Starting Real-Time Object Detection GUI...")
        print("📋 If GUI doesn't appear, check:")
        print("   • Camera permissions")
        print("   • Display/graphics drivers")
        print("   • Antivirus software blocking GUI")

        from gui import run_gui

        # Run the GUI - this will block until GUI is closed
        result = run_gui()

        # Ensure proper cleanup before printing final messages
        import time
        time.sleep(0.1)  # Small delay to ensure GUI cleanup is complete

        if result == 0:
            print("✅ GUI closed successfully")
        else:
            print(f"⚠️ GUI exited with code: {result}")

        return True

    except ImportError as e:
        print(f"❌ Failed to import GUI modules: {e}")
        print("💡 Troubleshooting:")
        print("   • Install PyQt6: pip install PyQt6")
        print("   • Check Python path and virtual environment")
        return False

    except Exception as e:
        print(f"❌ Error running GUI: {e}")
        print("💡 Troubleshooting:")
        print("   • Try running: python -c \"from gui import run_gui; run_gui()\"")
        print("   • Check if camera is accessible")
        print("   • Try a different model (yolov8n.pt)")
        import traceback
        traceback.print_exc()
        return False

def run_cli_mode():
    """Run CLI mode"""
    try:
        from detector import RealTimeDetector
        from config import DetectionConfig

        print("🚀 Starting Real-Time Object Detection (CLI Mode)...")
        print("Press 'q' to quit")

        config = DetectionConfig()
        with RealTimeDetector(config) as detector:
            if detector.initialize_camera():
                detector.run_detection_loop(show_gui=True)
                print("✅ Detection completed successfully!")
            else:
                print("❌ Failed to initialize camera")
                print("Troubleshooting:")
                print("• Check if camera is connected and not in use")
                print("• Try running: python run_detection.py --gui")
                return False

    except ImportError as e:
        print(f"❌ Failed to import detection modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in CLI detection: {e}")
        return False
    return True

def run_test_mode():
    """Run system tests"""
    try:
        print("🧪 Running System Tests...")

        # Test imports
        print("📦 Testing imports...")
        import torch
        import cv2
        import numpy as np
        from ultralytics import YOLO
        print("✅ All imports successful")

        # Test camera
        print("📷 Testing camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera test successful - Frame size: {frame.shape}")
                cap.release()
            else:
                print("❌ Camera opened but failed to capture frame")
                cap.release()
                return False
        else:
            print("❌ Failed to open camera")
            return False

        # Test model loading
        print("🤖 Testing model loading...")
        model = YOLO('yolov8n.pt')
        print("✅ Model loaded successfully")

        print("🎉 All tests passed! System is ready.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_info():
    """Show system information"""
    print("📋 System Information")
    print("=" * 30)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")

    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("OpenCV not available")

    try:
        import psutil
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        print(f"CPU cores: {cpu_count}")
        print(f"CPU frequency: {cpu_freq.current if cpu_freq else 'Unknown'} MHz")
        print(f"Memory: {memory.total / 1024 / 1024 / 1024:.1f} GB")
    except ImportError:
        print("psutil not available")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python run_detection.py --gui                    # Start GUI mode
  python run_detection.py --cli                    # Start CLI mode
  python run_detection.py --test                   # Run system tests
  python run_detection.py --info                   # Show system info
        """
    )

    parser.add_argument('--gui', action='store_true',
                       help='Run GUI mode (default)')
    parser.add_argument('--cli', action='store_true',
                       help='Run CLI mode')
    parser.add_argument('--test', action='store_true',
                       help='Run system tests')
    parser.add_argument('--info', action='store_true',
                       help='Show system information')

    args = parser.parse_args()

    # Determine mode
    if args.test:
        mode = 'test'
    elif args.cli:
        mode = 'cli'
    elif args.info:
        mode = 'info'
    else:
        mode = 'gui'  # Default to GUI mode

    # Check requirements first
    if not check_requirements():
        print("❌ Cannot proceed without required packages.")
        return 1

    # Run selected mode
    print(f"🎯 Starting Object Detection System ({mode.upper()} mode)")

    if mode == 'gui':
        success = run_gui_mode()
    elif mode == 'cli':
        success = run_cli_mode()
    elif mode == 'test':
        success = run_test_mode()
    elif mode == 'info':
        show_info()
        success = True
    else:
        print(f"❌ Unknown mode: {mode}")
        return 1

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
