#!/usr/bin/env python3
"""
Configuration loader for MediaPipe ROS wrapper
"""

import os
import yaml
from typing import Dict, Any


class MPConfig:
    """Configuration loader and manager for MediaPipe ROS wrapper"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration loader
        
        Args:
            config_file: Path to config file. If None, uses default package config.
        """
        if config_file is None:
            # Get package directory
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_file = os.path.join(package_dir, 'config', 'mp_config.yaml')
        
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_file} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file {self.config_file}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            'queue_size': {
                'default': 1,
                'image': 1,
                'pose': 1,
                'marker': 1,
                'gesture': 1
            },
            'processing': {
                'frequency': 25.0,
                'gpu_enabled': False
            },
            'features': {
                'plot_hpe_pose': True,
                'detect_hands': False,
                'detect_gestures': False,
                'plot_marker': False,
                'get_hand_orientation': False
            },
            'camera': {
                'oak_topic': '/oak/rgb/image_raw',
                'usb_topic': '/camera1/image_raw'
            },
            'models': {
                'pose_model': 'models/pose_landmarker_full.task',
                'hand_model': 'models/hand_landmarker.task'
            },
            'debug': {
                'enable_image_debug': False,
                'log_processing_time': True
            }
        }
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Path to config value (e.g., 'queue_size.image')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_queue_size(self, topic_type: str = 'default') -> int:
        """Get queue size for specific topic type"""
        return self.get(f'queue_size.{topic_type}', self.get('queue_size.default', 1))
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.get(f'features.{feature}', False)
    
    def get_camera_topic(self, camera_type: str) -> str:
        """Get camera topic name"""
        topic_key = f'{camera_type}_topic'
        return self.get(f'camera.{topic_key}', f'/{camera_type}/image_raw')
    
    def get_model_path(self, model_type: str, package_dir: str = None) -> str:
        """Get full path to model file"""
        if package_dir is None:
            # Get the source package directory, not build directory
            current_file = os.path.abspath(__file__)
            # Go up from mp_wrapper_ros/mp_config.py to mp_wrapper_ros/
            package_dir = os.path.dirname(os.path.dirname(current_file))
        
        model_file = self.get(f'models.{model_type}', f'models/{model_type}.task')
        full_path = os.path.join(package_dir, model_file)
        
        # Verify the path exists
        if not os.path.exists(full_path):
            # Fallback: try absolute path from workspace
            workspace_path = '/root/uav_ws/src/mp_ros_wrapper'
            fallback_path = os.path.join(workspace_path, model_file)
            if os.path.exists(fallback_path):
                return fallback_path
        
        return full_path
    
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()
    
    def __repr__(self):
        """String representation of config"""
        return f"MPConfig(file='{self.config_file}', keys={list(self.config.keys())})"