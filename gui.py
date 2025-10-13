"""
Enhanced GUI Interface for Real-Time Object Detection
Professional-grade interface with advanced features
"""

import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QComboBox,
                           QSlider, QGroupBox, QMessageBox, QProgressBar,
                           QCheckBox, QSpinBox, QTabWidget, QTextEdit,
                           QSplitter, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QColor, QPalette
import time
from typing import Optional, List, Dict, Any

from detector import RealTimeDetector
from config import DetectionConfig

class DetectionWorker(QThread):
    """Enhanced worker thread for detection"""

    frame_ready = pyqtSignal(np.ndarray, list)  # frame, detections
    stats_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, detector: RealTimeDetector):
        super().__init__()
        self.detector = detector
        self.running = False

    def run(self):
        """Enhanced detection loop"""
        self.running = True
        frame_count = 0
        
        # Initialize detector state
        self.detector.running = True
        self.detector.frame_count = 0
        self.detector.start_time = time.time()

        try:
            while self.running and self.detector.running:
                # Capture frame
                ret, frame = self.detector.cap.read()
                if not ret:
                    self.error_occurred.emit("Failed to capture frame from camera")
                    break

                # Detect objects
                annotated_frame, detections = self.detector.detect_objects(frame)
                
                # Increment detector frame count
                self.detector.frame_count += 1

                # Emit signals with detection data
                self.frame_ready.emit(annotated_frame, detections)

                # Update stats more frequently for real-time monitoring
                frame_count += 1
                if frame_count % 10 == 0:  # Update every 10 frames instead of 30
                    stats = self.detector.get_performance_stats()
                    self.stats_updated.emit(stats)

                # Optimized delay
                self.msleep(1)

        except Exception as e:
            self.error_occurred.emit(f"Detection error: {str(e)}")

    def stop(self):
        """Stop the worker thread"""
        self.running = False
        self.wait()

class ObjectInfoPanel(QFrame):
    """Panel to display detected objects information"""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setMaximumHeight(200)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🎯 Detected Objects")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        # Objects list
        self.objects_text = QTextEdit()
        self.objects_text.setMaximumHeight(120)
        self.objects_text.setHtml("<i>No objects detected yet...</i>")
        layout.addWidget(self.objects_text)

        # Count label
        self.count_label = QLabel("Count: 0")
        self.count_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.count_label)

class PerformancePanel(QFrame):
    """Enhanced performance monitoring panel"""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setMaximumWidth(300)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("📊 Performance Monitor")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)

        # FPS section
        fps_layout = QVBoxLayout()
        fps_title = QLabel("🚀 FPS:")
        fps_title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.fps_label = QLabel("0.0")
        self.fps_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.fps_label.setStyleSheet("color: #4CAF50;")
        fps_layout.addWidget(fps_title)
        fps_layout.addWidget(self.fps_label)

        # Progress bar
        self.fps_progress = QProgressBar()
        self.fps_progress.setRange(0, 60)
        fps_layout.addWidget(self.fps_progress)
        layout.addLayout(fps_layout)

        # System stats
        stats_layout = QVBoxLayout()

        # CPU
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("🖥️ CPU:"))
        self.cpu_label = QLabel("0%")
        cpu_layout.addWidget(self.cpu_label)
        cpu_layout.addStretch()
        stats_layout.addLayout(cpu_layout)

        # Memory
        mem_layout = QHBoxLayout()
        mem_layout.addWidget(QLabel("🧠 Memory:"))
        self.memory_label = QLabel("0%")
        mem_layout.addWidget(self.memory_label)
        mem_layout.addStretch()
        stats_layout.addLayout(mem_layout)

        # Frames
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("📸 Frames:"))
        self.frames_label = QLabel("0")
        frames_layout.addWidget(self.frames_label)
        frames_layout.addStretch()
        stats_layout.addLayout(frames_layout)

        layout.addLayout(stats_layout)

        # GPU Stats Section (RTX 2050)
        gpu_group = QGroupBox("🎮 GPU Monitor (RTX 2050)")
        gpu_layout = QVBoxLayout()
        
        # GPU Usage
        gpu_usage_layout = QHBoxLayout()
        gpu_usage_layout.addWidget(QLabel("⚡ Usage:"))
        self.gpu_usage_label = QLabel("0%")
        self.gpu_usage_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        gpu_usage_layout.addWidget(self.gpu_usage_label)
        gpu_usage_layout.addStretch()
        gpu_layout.addLayout(gpu_usage_layout)
        
        # GPU Usage Progress Bar
        self.gpu_usage_progress = QProgressBar()
        self.gpu_usage_progress.setRange(0, 100)
        self.gpu_usage_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3a3a3a;
                border-radius: 5px;
                text-align: center;
                background-color: #1f1f1f;
            }
            QProgressBar::chunk {
                background-color: #FF9800;
            }
        """)
        gpu_layout.addWidget(self.gpu_usage_progress)
        
        # GPU Memory
        gpu_mem_layout = QHBoxLayout()
        gpu_mem_layout.addWidget(QLabel("💾 Memory:"))
        self.gpu_memory_label = QLabel("0 / 4 GB")
        self.gpu_memory_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        gpu_mem_layout.addWidget(self.gpu_memory_label)
        gpu_mem_layout.addStretch()
        gpu_layout.addLayout(gpu_mem_layout)
        
        # GPU Memory Progress Bar
        self.gpu_memory_progress = QProgressBar()
        self.gpu_memory_progress.setRange(0, 4096)  # 4GB in MB
        self.gpu_memory_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3a3a3a;
                border-radius: 5px;
                text-align: center;
                background-color: #1f1f1f;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        gpu_layout.addWidget(self.gpu_memory_progress)
        
        # GPU Temperature
        gpu_temp_layout = QHBoxLayout()
        gpu_temp_layout.addWidget(QLabel("🌡️ Temp:"))
        self.gpu_temp_label = QLabel("--°C")
        self.gpu_temp_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        gpu_temp_layout.addWidget(self.gpu_temp_label)
        gpu_temp_layout.addStretch()
        gpu_layout.addLayout(gpu_temp_layout)
        
        # GPU Power
        gpu_power_layout = QHBoxLayout()
        gpu_power_layout.addWidget(QLabel("⚡ Power:"))
        self.gpu_power_label = QLabel("-- W")
        self.gpu_power_label.setStyleSheet("color: #FFEB3B; font-weight: bold;")
        gpu_power_layout.addWidget(self.gpu_power_label)
        gpu_power_layout.addStretch()
        gpu_layout.addLayout(gpu_power_layout)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)

class MainWindow(QMainWindow):
    """Enhanced main GUI window"""

    def __init__(self):
        super().__init__()
        self.detector = None
        self.worker = None
        self.config = DetectionConfig()
        self.detections_history = []
        self.max_history = 100

        # UI state
        self.detection_running = False
        
        # Initialize NVML for GPU monitoring
        self.nvml_available = False
        self.nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml_available = True
            print("✅ NVML initialized for GPU monitoring")
        except Exception as e:
            print(f"⚠️  NVML not available - Using PyTorch-only GPU monitoring")
            print(f"   (Reason: {str(e)[:50]})")
        
        # Stats update timer for reliable performance monitoring
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats_periodically)
        self.stats_timer.setInterval(500)  # Update every 500ms
        
        # GPU monitoring timer (independent of detection)
        self.gpu_timer = QTimer()
        self.gpu_timer.timeout.connect(self.update_gpu_stats_only)
        self.gpu_timer.setInterval(500)  # Update every 500ms

        self.init_ui()
        self.apply_modern_theme()
        
        # Start GPU monitoring immediately
        self.gpu_timer.start()
        print("🎮 GPU monitoring started")

    def init_ui(self):
        """Initialize enhanced user interface"""
        self.setWindowTitle("🔍 AI Object Detection System")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout with splitter for resizable panels
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Controls and info
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Center panel - Video display
        self.video_panel = self.create_video_panel()
        main_layout.addWidget(self.video_panel, 3)

        # Right panel - Performance and objects
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("🎮 Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Model selection
        model_group = QGroupBox("🤖 AI Model")
        model_layout = QVBoxLayout()

        model_layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'yolov8n.pt - Fastest (30-50 FPS)',
            'yolov8s.pt - Balanced ⭐ (25-40 FPS)',
            'yolov8m.pt - Accurate (20-35 FPS)',
            'yolov8l.pt - More Accurate (15-30 FPS)',
            'yolov8x.pt - Most Accurate (10-25 FPS)',
            'yolov9c.pt - YOLOv9 Compact (20-35 FPS)',
            'yolov9e.pt - YOLOv9 Extended ⭐⭐ (15-28 FPS)',
            'yolov10n.pt - YOLOv10 Nano (35-55 FPS)',
            'yolov10s.pt - YOLOv10 Small (30-45 FPS)',
            'yolov10m.pt - YOLOv10 Medium ⭐⭐ (25-40 FPS)',
            'yolov10l.pt - YOLOv10 Large (20-35 FPS)',
            'yolov10x.pt - YOLOv10 XLarge ⭐⭐⭐ (15-30 FPS)'
        ])
        self.model_combo.setCurrentIndex(1)  # Default to yolov8s
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.model_combo.setToolTip("Select AI model - RTX 2050 optimized!\n⭐ = Recommended | FPS estimates with GPU+FP16")
        model_layout.addWidget(self.model_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Detection settings
        settings_group = QGroupBox("⚙️ Detection Settings")
        settings_layout = QVBoxLayout()

        # Accuracy Mode
        accuracy_layout = QHBoxLayout()
        accuracy_layout.addWidget(QLabel("Accuracy Mode:"))
        self.accuracy_combo = QComboBox()
        self.accuracy_combo.addItems(['⚡ Fast', '⚖️ Balanced', '🎯 Accurate'])
        self.accuracy_combo.setCurrentIndex(1)  # Default to Balanced
        self.accuracy_combo.currentIndexChanged.connect(self.on_accuracy_mode_changed)
        self.accuracy_combo.setToolTip("Fast: Quick (conf 50%), Balanced: Good (conf 35%), Accurate: Best (conf 25%, 1280px)")
        accuracy_layout.addWidget(self.accuracy_combo)
        settings_layout.addLayout(accuracy_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(35)  # Updated default to match config
        self.conf_label = QLabel("35%")
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)
        self.conf_slider.setToolTip("Adjust confidence threshold in real-time (10-90%)\nLower = more detections, Higher = fewer but more confident")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        settings_layout.addLayout(conf_layout)

        # Camera source
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(['Camera 0', 'Camera 1', 'Camera 2'])
        self.cam_combo.setCurrentIndex(0)
        self.cam_combo.currentIndexChanged.connect(self.on_camera_changed)
        self.cam_combo.setToolTip("Switch camera source in real-time")
        cam_layout.addWidget(self.cam_combo)
        settings_layout.addLayout(cam_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶️ Start Detection")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.clicked.connect(self.toggle_detection)
        buttons_layout.addWidget(self.start_btn)

        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.setMinimumHeight(45)
        self.settings_btn.clicked.connect(self.show_settings)
        buttons_layout.addWidget(self.settings_btn)

        layout.addLayout(buttons_layout)

        # Status
        self.status_label = QLabel("Ready to start detection")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-weight: bold;
                padding: 10px;
                border: 1px solid #4CAF50;
                border-radius: 5px;
                background-color: rgba(76, 175, 80, 0.1);
            }
        """)
        layout.addWidget(self.status_label)

        layout.addStretch()
        return panel

    def create_video_panel(self):
        """Create video display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Video display
        self.video_label = QLabel("🎯 Ready for Object Detection\nClick 'Start' to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 3px solid #3a3a3a;
                border-radius: 10px;
                background-color: #1a1a1a;
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.video_label)

        # Detection info overlay
        self.overlay_label = QLabel("")
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.overlay_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 0.7);
                padding: 8px;
                border-radius: 5px;
            }
        """)
        # Position overlay on top of video
        self.overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.overlay_label)

        return panel

    def create_right_panel(self):
        """Create right information panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tab widget for different views
        self.tab_widget = QTabWidget()

        # Performance tab
        perf_tab = PerformancePanel()
        self.performance_panel = perf_tab
        self.tab_widget.addTab(perf_tab, "📊 Performance")

        # Objects tab
        objects_tab = ObjectInfoPanel()
        self.objects_panel = objects_tab
        self.tab_widget.addTab(objects_tab, "🎯 Objects")

        layout.addWidget(self.tab_widget)

        # Detection log
        log_group = QGroupBox("📝 Detection Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setHtml("<i>Detection events will appear here...</i>")
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        return panel

    def apply_modern_theme(self):
        """Apply enhanced modern theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f0f;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 1ex;
                background-color: #1f1f1f;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #4CAF50;
                background-color: #1f1f1f;
            }
            QPushButton {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                padding: 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #66BB6A;
            }
            QPushButton:pressed {
                background-color: #1f1f1f;
            }
            QComboBox, QSlider {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                padding: 8px;
            }
            QTabWidget::pane {
                border: 2px solid #3a3a3a;
                background-color: #1f1f1f;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #4CAF50;
                padding: 10px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: #ffffff;
            }
        """)

    def update_conf_label(self, value):
        """Update confidence label"""
        self.conf_label.setText(f"{value}%")
    
    def on_model_changed(self, index):
        """Handle model selection change"""
        model_names = [
            'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
            'yolov9c.pt', 'yolov9e.pt',
            'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt'
        ]
        new_model = model_names[index]
        
        if self.detection_running:
            reply = QMessageBox.question(
                self, 
                "Change Model",
                f"Changing the model requires restarting detection.\n\nSwitch to {new_model}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Store current camera to restore after restart
                current_camera = self.detector.current_camera_index if self.detector else 0
                
                self.status_label.setText(f"🔄 Switching to {new_model}...")
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #FF9800;
                        font-weight: bold;
                        padding: 10px;
                        border: 1px solid #FF9800;
                        border-radius: 5px;
                        background-color: rgba(255, 152, 0, 0.1);
                    }
                """)
                QApplication.processEvents()
                
                # Stop detection and clean up
                self.stop_detection()
                
                # Update config with new model BEFORE creating detector
                self.config.MODEL['name'] = new_model
                
                # Restart detection with new model
                if current_camera != 0:
                    self.cam_combo.setCurrentIndex(current_camera)
                self.start_detection()
                
                self.log_message(f"🔄 Model switched to {new_model} successfully")
            else:
                # Revert to previous selection
                model_map = {
                    'yolov8n.pt': 0, 'yolov8s.pt': 1, 'yolov8m.pt': 2, 'yolov8l.pt': 3, 'yolov8x.pt': 4,
                    'yolov9c.pt': 5, 'yolov9e.pt': 6,
                    'yolov10n.pt': 7, 'yolov10s.pt': 8, 'yolov10m.pt': 9, 'yolov10l.pt': 10, 'yolov10x.pt': 11
                }
                if self.detector and self.detector.config.MODEL['name'] in model_map:
                    self.model_combo.blockSignals(True)
                    self.model_combo.setCurrentIndex(model_map[self.detector.config.MODEL['name']])
                    self.model_combo.blockSignals(False)
        else:
            # Not running - just update the config
            self.config.MODEL['name'] = new_model
            self.log_message(f"📝 Model set to: {new_model}")
    
    def on_accuracy_mode_changed(self, index):
        """Handle accuracy mode change"""
        modes = ['fast', 'balanced', 'accurate']
        mode_names = ['⚡ Fast', '⚖️ Balanced', '🎯 Accurate']
        mode = modes[index]
        mode_name = mode_names[index]
        
        if self.detector:
            self.detector.set_accuracy_mode(mode)
            if self.detection_running:
                self.log_message(f"🔧 Accuracy mode changed to: {mode_name}")
            
            # Update confidence slider to match mode
            if mode == 'fast':
                self.conf_slider.blockSignals(True)
                self.conf_slider.setValue(50)
                self.conf_slider.blockSignals(False)
            elif mode == 'balanced':
                self.conf_slider.blockSignals(True)
                self.conf_slider.setValue(35)
                self.conf_slider.blockSignals(False)
            elif mode == 'accurate':
                self.conf_slider.blockSignals(True)
                self.conf_slider.setValue(25)
                self.conf_slider.blockSignals(False)
        else:
            self.log_message(f"📝 Accuracy mode set to: {mode_name}")
    
    def on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        new_threshold = value / 100.0
        if self.detector:
            self.detector.config.MODEL['confidence_threshold'] = new_threshold
            if self.detection_running:
                self.log_message(f"🎯 Confidence threshold updated to {value}%")
    
    def on_camera_changed(self, index):
        """Handle camera selection change"""
        if self.detection_running:
            new_camera_index = index
            if self.detector:
                self.status_label.setText(f"🔄 Switching to Camera {new_camera_index}...")
                QApplication.processEvents()
                
                success = self.detector.switch_camera(new_camera_index)
                
                if success:
                    self.log_message(f"📷 Switched to Camera {new_camera_index}")
                    self.status_label.setText("✅ Detection running - Live object detection active")
                else:
                    self.log_message(f"❌ Failed to switch to Camera {new_camera_index}")
                    QMessageBox.warning(
                        self,
                        "Camera Switch Failed",
                        f"Could not switch to Camera {new_camera_index}.\n\nThe camera may not be available."
                    )
                    # Revert to previous camera
                    if self.detector:
                        self.cam_combo.blockSignals(True)
                        self.cam_combo.setCurrentIndex(self.detector.current_camera_index)
                        self.cam_combo.blockSignals(False)
        else:
            self.log_message(f"📝 Camera set to: Camera {index}")

    def toggle_detection(self):
        """Start or stop detection"""
        if not self.detection_running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        """Start object detection with enhanced setup"""
        try:
            # Update status
            self.status_label.setText("🔄 Initializing detection system...")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #FF9800;
                    font-weight: bold;
                    padding: 10px;
                    border: 1px solid #FF9800;
                    border-radius: 5px;
                    background-color: rgba(255, 152, 0, 0.1);
                }
            """)
            QApplication.processEvents()

            # Initialize detector if needed
            if self.detector is None:
                # Update config from UI BEFORE creating detector
                model_names = [
                    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                    'yolov9c.pt', 'yolov9e.pt',
                    'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt'
                ]
                model_name = model_names[self.model_combo.currentIndex()]
                self.config.MODEL['name'] = model_name
                self.config.MODEL['confidence_threshold'] = self.conf_slider.value() / 100.0
                
                # Now create detector with correct config
                self.detector = RealTimeDetector(self.config)
                print(f"✅ Detector created with model: {self.detector.config.MODEL['name']}")

            # Initialize camera with selected source
            camera_source = self.cam_combo.currentIndex()
            if not self.detector.initialize_camera(camera_source):
                QMessageBox.critical(self, "Camera Error",
                                   f"Failed to initialize camera {camera_source}.\n\n"
                                   "Please check:\n"
                                   "• Camera is connected and powered on\n"
                                   "• Camera permissions are granted\n"
                                   "• No other application is using the camera\n"
                                   "• Try a different camera (0, 1, 2)")
                return

            # Start worker thread
            self.worker = DetectionWorker(self.detector)
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.stats_updated.connect(self.update_stats)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.start()
            
            # Emit initial stats immediately
            stats = self.detector.get_performance_stats()
            self.update_stats(stats)
            
            # Start periodic stats timer
            self.stats_timer.start()
            
            # Update camera combo to reflect actual camera being used
            self.cam_combo.blockSignals(True)
            self.cam_combo.setCurrentIndex(self.detector.current_camera_index)
            self.cam_combo.blockSignals(False)
            
            # Update model combo to reflect actual model being used
            model_map = {
                'yolov8n.pt': 0, 'yolov8s.pt': 1, 'yolov8m.pt': 2, 'yolov8l.pt': 3, 'yolov8x.pt': 4,
                'yolov9c.pt': 5, 'yolov9e.pt': 6,
                'yolov10n.pt': 7, 'yolov10s.pt': 8, 'yolov10m.pt': 9, 'yolov10l.pt': 10, 'yolov10x.pt': 11
            }
            if self.detector.config.MODEL['name'] in model_map:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(model_map[self.detector.config.MODEL['name']])
                self.model_combo.blockSignals(False)

            # Update UI
            self.detection_running = True
            self.start_btn.setText("⏹️ Stop Detection")
            self.status_label.setText("✅ Detection running - Live object detection active")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-weight: bold;
                    padding: 10px;
                    border: 1px solid #4CAF50;
                    border-radius: 5px;
                    background-color: rgba(76, 175, 80, 0.1);
                }
            """)

            self.log_message("🚀 Detection started successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start detection: {str(e)}")
            self.log_message(f"❌ Error: {str(e)}")

    def stop_detection(self):
        """Stop object detection"""
        try:
            # Stop stats timer
            self.stats_timer.stop()
            
            # Stop worker
            if self.worker:
                self.worker.stop()
                self.worker = None

            # Cleanup detector
            if self.detector:
                self.detector.cleanup()
                self.detector = None

            # Update UI
            self.detection_running = False
            self.start_btn.setText("▶️ Start Detection")
            self.video_label.setText("⏹️ Detection stopped\nClick 'Start' to begin again")
            self.overlay_label.setText("")
            self.status_label.setText("Ready to start detection")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-weight: bold;
                    padding: 10px;
                    border: 1px solid #4CAF50;
                    border-radius: 5px;
                    background-color: rgba(76, 175, 80, 0.1);
                }
            """)

            self.log_message("⏹️ Detection stopped")

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error stopping detection: {str(e)}")

    def update_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """Update video display with enhanced visualization"""
        if frame is not None:
            # Convert BGR to RGB for PyQt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create QImage correctly for PyQt6
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            # Use copy() to ensure data persists after the function returns
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(qt_image)

            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            self.video_label.setPixmap(scaled_pixmap)

            # Update overlay with detection info
            if detections:
                overlay_text = f"🎯 Objects: {len(detections)}"
                # Show top detections
                top_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:3]
                overlay_text += "\n" + "\n".join([
                    f"• {d['class_name']}: {d['confidence']:.2f}"
                    for d in top_detections
                ])
                self.overlay_label.setText(overlay_text)
            else:
                self.overlay_label.setText("🎯 No objects detected")

            # Update objects panel
            self.update_objects_panel(detections)

    def update_stats(self, stats: dict):
        """Update performance statistics"""
        try:
            fps = stats.get('fps', 0)
            self.performance_panel.fps_label.setText(f"{fps:.1f}")
            self.performance_panel.fps_progress.setValue(int(min(fps, 60)))

            self.performance_panel.cpu_label.setText(f"{stats.get('cpu_usage', 0):.1f}%")
            self.performance_panel.memory_label.setText(f"{stats.get('memory_usage', 0):.1f}%")
            self.performance_panel.frames_label.setText(f"{stats.get('frames_processed', 0)}")
            
            # Update GPU stats
            gpu_stats = self.get_gpu_stats()
            if gpu_stats:
                # GPU Usage
                gpu_usage = gpu_stats.get('gpu_usage', 0)
                self.performance_panel.gpu_usage_label.setText(f"{gpu_usage:.1f}%")
                self.performance_panel.gpu_usage_progress.setValue(int(gpu_usage))
                
                # GPU Memory
                gpu_mem_used = gpu_stats.get('gpu_memory_used', 0)
                gpu_mem_total = gpu_stats.get('gpu_memory_total', 4096)
                self.performance_panel.gpu_memory_label.setText(f"{gpu_mem_used / 1024:.2f} / {gpu_mem_total / 1024:.1f} GB")
                self.performance_panel.gpu_memory_progress.setValue(int(gpu_mem_used))
                
                # GPU Temperature
                gpu_temp = gpu_stats.get('gpu_temp', 0)
                if gpu_temp > 0:
                    temp_color = "#4CAF50"  # Green
                    if gpu_temp > 70:
                        temp_color = "#FF9800"  # Orange
                    if gpu_temp > 85:
                        temp_color = "#F44336"  # Red
                    self.performance_panel.gpu_temp_label.setStyleSheet(f"color: {temp_color}; font-weight: bold;")
                    self.performance_panel.gpu_temp_label.setText(f"{gpu_temp:.0f}°C")
                
                # GPU Power
                gpu_power = gpu_stats.get('gpu_power', 0)
                if gpu_power > 0:
                    self.performance_panel.gpu_power_label.setText(f"{gpu_power:.1f} W")
                    
        except (AttributeError, KeyError, ValueError) as e:
            # Handle cases where performance panel might not be initialized
            pass
    
    def update_stats_periodically(self):
        """Periodically update stats via timer"""
        if self.detection_running and self.detector:
            try:
                stats = self.detector.get_performance_stats()
                self.update_stats(stats)
            except Exception as e:
                pass  # Silently handle errors during periodic updates
    
    def update_gpu_stats_only(self):
        """Update GPU stats independently (always running)"""
        try:
            gpu_stats = self.get_gpu_stats()
            if gpu_stats and hasattr(self, 'performance_panel'):
                # GPU Usage
                gpu_usage = gpu_stats.get('gpu_usage', 0)
                self.performance_panel.gpu_usage_label.setText(f"{gpu_usage:.1f}%")
                self.performance_panel.gpu_usage_progress.setValue(int(gpu_usage))
                
                # GPU Memory
                gpu_mem_used = gpu_stats.get('gpu_memory_used', 0)
                gpu_mem_total = gpu_stats.get('gpu_memory_total', 4096)
                self.performance_panel.gpu_memory_label.setText(f"{gpu_mem_used / 1024:.2f} / {gpu_mem_total / 1024:.1f} GB")
                self.performance_panel.gpu_memory_progress.setValue(int(gpu_mem_used))
                
                # GPU Temperature
                gpu_temp = gpu_stats.get('gpu_temp', 0)
                if gpu_temp > 0:
                    temp_color = "#4CAF50"  # Green
                    if gpu_temp > 70:
                        temp_color = "#FF9800"  # Orange
                    if gpu_temp > 85:
                        temp_color = "#F44336"  # Red
                    self.performance_panel.gpu_temp_label.setStyleSheet(f"color: {temp_color}; font-weight: bold;")
                    self.performance_panel.gpu_temp_label.setText(f"{gpu_temp:.0f}°C")
                else:
                    self.performance_panel.gpu_temp_label.setText("--°C")
                
                # GPU Power
                gpu_power = gpu_stats.get('gpu_power', 0)
                if gpu_power > 0:
                    self.performance_panel.gpu_power_label.setText(f"{gpu_power:.1f} W")
                else:
                    self.performance_panel.gpu_power_label.setText("-- W")
        except Exception as e:
            pass  # Silently handle errors

    def update_objects_panel(self, detections: List[Dict[str, Any]]):
        """Update objects information panel"""
        try:
            if not detections:
                self.objects_panel.objects_text.setHtml("<i>No objects detected...</i>")
                self.objects_panel.count_label.setText("Count: 0")
                return

            # Group detections by class
            class_counts = {}
            for detection in detections:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Create HTML content
            html_content = "<b>Detected Objects:</b><br>"
            for class_name, count in sorted(class_counts.items()):
                html_content += f"• {class_name}: {count}<br>"

            self.objects_panel.objects_text.setHtml(html_content)
            self.objects_panel.count_label.setText(f"Count: {len(detections)}")

            # Add to history for analysis
            self.detections_history.append({
                'timestamp': time.time(),
                'detections': detections,
                'count': len(detections)
            })

            # Keep history manageable
            if len(self.detections_history) > self.max_history:
                self.detections_history = self.detections_history[-self.max_history:]

        except (AttributeError, KeyError, ValueError) as e:
            # Handle cases where objects panel might not be initialized
            pass

    def log_message(self, message: str):
        """Add message to detection log"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}<br>"

            current_html = self.log_text.toHtml()
            if "<i>Detection events will appear here...</i>" in current_html:
                new_html = log_entry
            else:
                new_html = current_html + log_entry

            self.log_text.setHtml(new_html)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        except (AttributeError, ValueError) as e:
            # Handle cases where log_text might not be initialized
            pass

    def handle_error(self, error_msg: str):
        """Handle errors from worker thread"""
        QMessageBox.critical(self, "Detection Error", error_msg)
        self.log_message(f"❌ Error: {error_msg}")
        self.stop_detection()

    def show_settings(self):
        """Show advanced settings dialog"""
        QMessageBox.information(self, "Settings",
                               "Advanced settings will be available in the next version.\n\n"
                               "Current settings:\n"
                               "• Model updates in real-time\n"
                               "• Confidence threshold adjustable\n"
                               "• Multiple camera support")

    def get_gpu_stats(self):
        """Get real-time GPU statistics for RTX 2050"""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return None
            
            stats = {}
            
            # GPU Memory (using PyTorch - always works)
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)  # MB
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
            
            stats['gpu_memory_used'] = gpu_mem_reserved  # Use reserved as it's more accurate
            stats['gpu_memory_total'] = gpu_mem_total
            
            # Try to get GPU usage, temp, and power using pynvml
            if self.nvml_available and self.nvml_handle:
                try:
                    import pynvml
                    
                    # GPU Utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    stats['gpu_usage'] = utilization.gpu
                    
                    # GPU Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                        stats['gpu_temp'] = temp
                    except:
                        stats['gpu_temp'] = 0
                    
                    # GPU Power
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0  # Convert mW to W
                        stats['gpu_power'] = power
                    except:
                        stats['gpu_power'] = 0
                        
                except Exception as e:
                    # Fallback to estimation
                    mem_usage_percent = (gpu_mem_reserved / gpu_mem_total) * 100
                    stats['gpu_usage'] = min(mem_usage_percent * 2, 100)
                    stats['gpu_temp'] = 0
                    stats['gpu_power'] = 0
            else:
                # pynvml not available, estimate GPU usage from memory + recent activity
                mem_usage_percent = (gpu_mem_reserved / gpu_mem_total) * 100
                
                # More realistic estimation: if memory is being used, GPU is probably active
                if gpu_mem_reserved > 100:  # More than 100MB reserved means model loaded
                    # Estimate based on memory pressure and assume some utilization
                    base_usage = min(mem_usage_percent * 2, 60)
                    # If detection is running, assume higher usage
                    if self.detection_running:
                        stats['gpu_usage'] = min(base_usage + 20, 95)
                    else:
                        stats['gpu_usage'] = base_usage
                else:
                    stats['gpu_usage'] = mem_usage_percent
                
                stats['gpu_temp'] = 0  # Not available without NVML
                stats['gpu_power'] = 0  # Not available without NVML
            
            return stats
            
        except Exception as e:
            return None
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.detection_running:
            self.stop_detection()
        event.accept()

def run_gui():
    """Run the enhanced GUI application"""
    try:
        print("🚀 Initializing Enhanced Object Detection GUI...")

        # Create QApplication instance
        app = QApplication(sys.argv)
        print("✅ QApplication created")

        # Set application properties
        app.setApplicationName("AI Object Detection System")
        app.setApplicationVersion("3.0.0")
        app.setOrganizationName("AI Assistant")

        # Set modern Fusion style
        app.setStyle('Fusion')
        print("✅ Modern theme applied")

        # Create main window
        window = MainWindow()
        print("✅ Enhanced main window created")

        # Show the window
        window.show()
        print("✅ GUI window displayed")

        # Ensure window is raised and activated
        window.raise_()
        window.activateWindow()

        print("🎯 Enhanced GUI ready!")
        print("📋 Features available:")
        print("   • Real-time object detection")
        print("   • Performance monitoring")
        print("   • Live detection logging")
        print("   • Multiple AI model support")
        print("   • Configurable confidence thresholds")

        # Run the application event loop
        result = app.exec()

        # Only print exit code if it's not 0 (success)
        if result != 0:
            print(f"⚠️ GUI exited with code: {result}")

        return result

    except Exception as e:
        print(f"❌ Failed to start Enhanced GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_gui())
