"""
Simplified configuration for Real-Time Object Detection System
"""

import os
from typing import Dict, Any

class DetectionConfig:
    """Essential configuration for object detection"""

    # Model settings
    MODEL = {
        'name': 'yolov8s.pt',  # Balanced model - good for RTX 2050
        'confidence_threshold': 0.35,  # Lowered for better detection (was 0.5)
        'iou_threshold': 0.45,  # Non-Maximum Suppression threshold
        'input_size': 640,  # Image size for inference (640, 800, or 1280)
        'max_detections': 300,  # Maximum detections per image
        'agnostic_nms': False,  # Class-agnostic NMS
        'half_precision': True,  # Use FP16 for 2x speed on RTX 2050!
        'device': 'auto',  # auto, cuda, cpu - auto detects GPU
        'augment': False,  # Test-time augmentation for higher accuracy
        'enhance_contrast': False,  # Apply contrast enhancement preprocessing
        'denoise': False,  # Apply denoising before inference
        'min_box_area': 0,  # Filter out very small detections (pixels)
        'class_filter': None,  # Limit detection to specific classes (list of IDs)
        'fuse_model': True,  # Fuse model layers for better performance/precision
    }

    # Camera settings
    CAMERA = {
        'source': 0,  # Default webcam
        'width': 1280,
        'height': 720,
        'fps': 30,
    }

    # GUI settings
    GUI = {
        'window_title': 'Real-Time Object Detection',
        'window_width': 1280,
        'window_height': 720,
        'theme': 'dark'
    }

    # Performance settings
    PERFORMANCE = {
        'show_fps': True,
        'adaptive_resolution': False,
        'use_gpu': True,  # Enable GPU acceleration (RTX 2050)
        'gpu_optimization': True,  # Optimize for NVIDIA GPUs
    }

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return {
            'model': cls.MODEL,
            'camera': cls.CAMERA,
            'gui': cls.GUI,
            'performance': cls.PERFORMANCE
        }
