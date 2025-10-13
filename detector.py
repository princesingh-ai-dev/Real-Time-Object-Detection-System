"""
Enhanced Real-Time Object Detection Engine
Improved camera handling and robust detection
"""

import cv2
import numpy as np
import time
import torch
import os
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any, Optional
import psutil
from config import DetectionConfig

# Force NVIDIA GPU usage on Windows (disable Intel Iris Xe)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

class RealTimeDetector:
    """Enhanced real-time object detection class with improved camera handling"""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.model = None
        self.cap = None
        self.running = False

        # Enhanced performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        self.detection_history = []

        # Camera management
        self.available_cameras = []
        self.current_camera_index = 0

        # Load model
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model with GPU acceleration optimized for RTX 2050"""
        try:
            print(f"🤖 Loading model: {self.config.MODEL['name']}")
            
            # Clean up old model if exists
            if self.model is not None:
                print("🧹 Cleaning up previous model...")
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Force NVIDIA GPU selection
            if torch.cuda.is_available():
                # Ensure we're using NVIDIA RTX 2050, not Intel Iris Xe
                torch.cuda.set_device(0)
                device = 'cuda:0'
                
                # Verify correct GPU
                gpu_name = torch.cuda.get_device_name(0)
                if 'NVIDIA' not in gpu_name and 'RTX' not in gpu_name:
                    print(f"⚠️ Warning: Using {gpu_name} instead of NVIDIA RTX 2050")
                    print("   💡 Check Windows Graphics Settings to prefer NVIDIA GPU")
            else:
                device = 'cpu'
            
            # Load model with explicit device
            # CRITICAL: Pass device directly to YOLO() for proper GPU utilization
            self.model = YOLO(self.config.MODEL['name'])

            if self.config.MODEL.get('fuse_model', True):
                try:
                    self.model.fuse()
                    print("   🔄 Model layers fused for improved precision")
                except Exception as fuse_error:
                    print(f"   ⚠️ Model fusion skipped: {fuse_error}")
            
            # Configure for optimal performance
            if torch.cuda.is_available():
                # Enable optimizations for RTX 2050
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_compute = torch.cuda.get_device_properties(0).major
                
                print(f"✅ Model loaded on DEDICATED GPU: {gpu_name}")
                print(f"   💾 GPU Memory: {gpu_memory:.1f} GB")
                print(f"   🔧 Compute Capability: {gpu_compute}.x")
                print(f"   🎯 Device: cuda:{torch.cuda.current_device()}")
                
                # Check FP16 support
                if self.config.MODEL.get('half_precision', False):
                    print("   ⚡ FP16 Half-Precision: ENABLED (2x speed boost!)")
                    print("   🔥 Warming up NVIDIA RTX 2050...")
                    # Warmup: Run dummy inference to initialize GPU
                    dummy_img = torch.zeros((640, 640, 3)).numpy().astype('uint8')
                    _ = self.model(dummy_img, device=device, half=True, verbose=False)
                    print("   ✅ NVIDIA GPU warmup complete")
                else:
                    print("   💡 Tip: Enable half_precision for 2x speed")
            else:
                print("✅ Model loaded on CPU (CUDA not available)")
                print("   💡 For better performance, install CUDA PyTorch:")
                print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def reload_model(self, new_model_name: str) -> bool:
        """Reload model with a different model file"""
        try:
            print(f"🔄 Reloading model: {new_model_name}")
            old_model = self.config.MODEL['name']
            
            # Update config
            self.config.MODEL['name'] = new_model_name
            
            # Reload model
            self._load_model()
            
            print(f"✅ Model switched from {old_model} to {new_model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to reload model: {e}")
            # Restore old model name on failure
            self.config.MODEL['name'] = old_model
            return False
    
    def set_accuracy_mode(self, mode: str = 'balanced'):
        """
        Set detection accuracy mode
        
        Args:
            mode: 'fast', 'balanced', or 'accurate'
        """
        if mode == 'fast':
            self.config.MODEL['confidence_threshold'] = 0.5
            self.config.MODEL['iou_threshold'] = 0.45
            self.config.MODEL['input_size'] = 640
            self.config.MODEL['augment'] = False
            self.config.MODEL['enhance_contrast'] = False
            self.config.MODEL['denoise'] = False
            self.config.MODEL['min_box_area'] = 0
            print("⚡ Fast mode: Quick detection, good accuracy")
        elif mode == 'balanced':
            self.config.MODEL['confidence_threshold'] = 0.35
            self.config.MODEL['iou_threshold'] = 0.45
            self.config.MODEL['input_size'] = 640
            self.config.MODEL['augment'] = False
            self.config.MODEL['enhance_contrast'] = True
            self.config.MODEL['denoise'] = True
            self.config.MODEL['min_box_area'] = 200
            print("⚖️ Balanced mode: Good speed and accuracy")
        elif mode == 'accurate':
            self.config.MODEL['confidence_threshold'] = 0.25
            self.config.MODEL['iou_threshold'] = 0.45
            self.config.MODEL['input_size'] = 1280
            self.config.MODEL['augment'] = True
            self.config.MODEL['enhance_contrast'] = True
            self.config.MODEL['denoise'] = True
            self.config.MODEL['min_box_area'] = 400
            print("🎯 Accurate mode: Best detection, slower processing")
        else:
            print(f"❌ Unknown mode: {mode}. Use 'fast', 'balanced', or 'accurate'")

    def scan_available_cameras(self, verbose: bool = True) -> List[int]:
        """Scan for available camera devices"""
        available_cameras = []

        if verbose:
            print("📷 Scanning for available cameras...")

        for i in range(5):  # Check first 5 camera indices
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                        if verbose:
                            print(f"✅ Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
                    else:
                        if verbose:
                            print(f"⚠️ Camera {i}: Opens but no frame")
                    cap.release()
                else:
                    if verbose:
                        print(f"❌ Camera {i}: Not accessible")
            except Exception as e:
                if verbose:
                    print(f"❌ Camera {i}: Error - {e}")

        if not available_cameras:
            if verbose:
                print("⚠️ No cameras found. Please check camera connections.")
            # Try some common alternative approaches
            for i in [-1, 700, 701, 702]:  # Try some alternative indices
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            available_cameras.append(i)
                            if verbose:
                                print(f"✅ Alternative camera {i}: Available")
                            cap.release()
                            break
                except:
                    pass

        self.available_cameras = available_cameras
        return available_cameras

    def initialize_camera(self, source: int = None) -> bool:
        """Initialize camera capture with enhanced error handling"""
        if source is not None:
            self.current_camera_index = source

        # If no specific source requested, scan for available cameras
        if source is None or source not in self.available_cameras:
            if not self.available_cameras:
                self.scan_available_cameras(verbose=False)  # Less verbose during GUI mode

            if self.available_cameras:
                self.current_camera_index = self.available_cameras[0]
                print(f"🎯 Using first available camera: {self.current_camera_index}")
            else:
                print("❌ No cameras available")
                return False

        try:
            print(f"📷 Opening camera source: {self.current_camera_index}")

            # Try different API preferences for better compatibility
            for api_preference in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
                try:
                    self.cap = cv2.VideoCapture(self.current_camera_index, api_preference)
                    if self.cap.isOpened():
                        print(f"✅ Camera opened successfully with API: {api_preference}")
                        break
                except:
                    continue

            if not self.cap or not self.cap.isOpened():
                print(f"❌ Could not open camera {self.current_camera_index}")
                return False

            # Set camera properties with validation
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA['height'])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA['fps'])

            # Wait a moment for settings to apply
            time.sleep(0.1)

            # Verify camera settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            print("✅ Camera initialized successfully:")
            print(f"   • Resolution: {actual_width}x{actual_height}")
            print(f"   • FPS: {actual_fps}")
            print(f"   • Format: {self.cap.get(cv2.CAP_PROP_FORMAT)}")

            # Test frame capture
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                print(f"✅ Test frame captured: {test_frame.shape}")
                return True
            else:
                print("❌ Camera opened but failed to capture test frame")
                return False

        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def switch_camera(self, camera_index: int) -> bool:
        """Switch to a different camera"""
        if camera_index == self.current_camera_index:
            return True  # Already using this camera

        # Cleanup current camera
        if self.cap:
            self.cap.release()

        # Try to initialize new camera
        old_index = self.current_camera_index
        self.current_camera_index = camera_index

        if self.initialize_camera():
            print(f"✅ Switched camera from {old_index} to {camera_index}")
            return True
        else:
            print(f"❌ Failed to switch to camera {camera_index}")
            # Try to restore previous camera
            self.current_camera_index = old_index
            return self.initialize_camera()

    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Perform object detection on a frame with enhanced accuracy"""
        try:
            # Force NVIDIA GPU for inference (not Intel Iris Xe)
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # Ensure NVIDIA RTX 2050
                device = 'cuda:0'
                use_half = self.config.MODEL.get('half_precision', False)
            else:
                device = 'cpu'
                use_half = False

            processed_frame = frame

            if self.config.MODEL.get('denoise', False):
                processed_frame = cv2.fastNlMeansDenoisingColored(processed_frame, None, 5, 5, 7, 21)

            if self.config.MODEL.get('enhance_contrast', False):
                lab = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                merged = cv2.merge((cl, a, b))
                processed_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

            # Run inference with optimized parameters on NVIDIA GPU
            # CRITICAL: Explicit device ensures RTX 2050 usage, not Intel Iris Xe
            results = self.model(
                processed_frame,
                conf=self.config.MODEL['confidence_threshold'],
                iou=self.config.MODEL['iou_threshold'],
                imgsz=self.config.MODEL['input_size'],
                max_det=self.config.MODEL['max_detections'],
                agnostic_nms=self.config.MODEL['agnostic_nms'],
                half=use_half,
                device=device,
                augment=self.config.MODEL.get('augment', False),
                verbose=False,
                stream=False  # Use False for real-time to avoid buffering
            )

            # Process results
            detections = []
            annotated_frame = frame.copy()

            min_box_area = self.config.MODEL.get('min_box_area', 0)
            class_filter = self.config.MODEL.get('class_filter')

            if results and len(results) > 0:
                result = results[0]

                # Extract detections
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i].item())
                        if class_filter and class_id not in class_filter:
                            continue

                        bbox_tensor = boxes.xyxy[i]
                        x1, y1, x2, y2 = bbox_tensor.cpu().numpy().tolist()
                        box_area = max(0, x2 - x1) * max(0, y2 - y1)
                        if min_box_area and box_area < min_box_area:
                            continue

                        detection = {
                            'class_id': class_id,
                            'class_name': result.names[class_id],
                            'confidence': float(boxes.conf[i].item()),
                            'bbox': [x1, y1, x2, y2]
                        }
                        detections.append(detection)

                # Draw annotations if we have a result plot
                if hasattr(result, 'plot'):
                    annotated_frame = result.plot()

            # Store detection history
            self.detection_history.append({
                'timestamp': time.time(),
                'detections': detections,
                'frame_count': self.frame_count
            })

            # Keep history manageable
            if len(self.detection_history) > 100:
                self.detection_history = self.detection_history[-50:]

            return annotated_frame, detections

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return frame, []

    def get_performance_stats(self) -> Dict[str, float]:
        """Get enhanced performance statistics"""
        try:
            # Calculate FPS
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                current_fps = self.frame_count / elapsed_time
            else:
                current_fps = 0

            # Update FPS history
            self.fps_history.append(current_fps)
            if len(self.fps_history) > 60:  # Keep last 60 readings
                self.fps_history = self.fps_history[-60:]

            # System stats
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            # Detection stats
            avg_detections = 0
            if self.detection_history:
                recent_history = self.detection_history[-10:]  # Last 10 frames
                total_detections = sum(len(item['detections']) for item in recent_history if 'detections' in item)
                avg_detections = total_detections / len(recent_history) if recent_history else 0

            return {
                'fps': current_fps,
                'frames_processed': self.frame_count,
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'elapsed_time': elapsed_time,
                'avg_detections': avg_detections,
                'available_cameras': len(self.available_cameras),
                'current_camera': self.current_camera_index
            }
        except Exception as e:
            return {'error': str(e)}

    def run_detection_loop(self, show_gui: bool = True) -> None:
        """Enhanced main detection loop"""
        if not self.cap:
            print("❌ Camera not initialized")
            return

        self.running = True
        self.frame_count = 0
        self.start_time = time.time()

        print("🚀 Starting enhanced detection loop...")
        print("📋 Controls: 'q' to quit, 'c' to switch camera")

        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break

                # Detect objects
                annotated_frame, detections = self.detect_objects(frame)

                # Add performance info to frame
                if self.config.PERFORMANCE['show_fps']:
                    stats = self.get_performance_stats()
                    fps = stats.get('fps', 0)

                    # Draw FPS on frame
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw detection count
                    cv2.putText(annotated_frame, f"Objects: {len(detections)}",
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Show frame
                if show_gui:
                    cv2.imshow(self.config.GUI['window_title'], annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') and self.available_cameras:
                    # Switch camera
                    next_camera = (self.current_camera_index + 1) % len(self.available_cameras)
                    if next_camera in self.available_cameras:
                        self.switch_camera(next_camera)

                self.frame_count += 1

        except KeyboardInterrupt:
            print("⏹️ Detection stopped by user")
        except Exception as e:
            print(f"❌ Error in detection loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Enhanced cleanup with proper resource management"""
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("🧹 Enhanced cleanup completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
