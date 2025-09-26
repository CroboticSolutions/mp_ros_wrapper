#!/usr/bin/env python3
"""
Test script for VitPose ROS wrapper
"""

import sys
import os

# Add the package path to sys.path for testing
sys.path.insert(0, '/root/uav_ws/src/mp_ros_wrapper')

def test_imports():
    """Test if all required imports work"""
    print("🧪 Testing VitPose ROS wrapper imports...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, infer_device
        print("✅ Transformers imports successful")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import supervision as sv
        print("✅ Supervision import successful")
    except ImportError as e:
        print(f"❌ Supervision import failed: {e}")
        print("💡 Install with: pip install supervision")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL import successful")
    except ImportError as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    try:
        import rclpy
        from sensor_msgs.msg import Image as RosImage
        from hpe_ros_msgs.msg import MpHumanPose3D
        print("✅ ROS 2 imports successful")
    except ImportError as e:
        print(f"❌ ROS 2 imports failed: {e}")
        return False
    
    try:
        from mp_wrapper_ros.vitpose_ros_wrapper import VitPoseROSWrapper
        print("✅ VitPose ROS wrapper import successful")
    except ImportError as e:
        print(f"❌ VitPose ROS wrapper import failed: {e}")
        return False
    
    return True

def test_device():
    """Test device detection"""
    print("\n🔍 Testing device detection...")
    try:
        from transformers import infer_device
        device = infer_device()
        print(f"✅ Detected device: {device}")
        return True
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded (this might take a while)"""
    print("\n📥 Testing model loading...")
    try:
        from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, infer_device
        
        device = infer_device()
        print(f"Using device: {device}")
        
        # Test person detection model
        print("Loading person detection model...")
        person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
        print("✅ Person detection model loaded successfully")
        
        # Test VitPose model
        print("Loading VitPose model...")
        pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)
        print("✅ VitPose model loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    print("🚀 VitPose ROS Wrapper Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing dependencies.")
        return False
    
    # Test device
    if not test_device():
        print("\n❌ Device detection failed.")
        return False
    
    # Test model loading (optional, can be slow)
    print("\n" + "=" * 50)
    print("⚠️  Model loading test (this may take several minutes)...")
    test_models = input("Do you want to test model loading? (y/N): ").lower().startswith('y')
    
    if test_models:
        if not test_model_loading():
            print("\n❌ Model loading tests failed.")
            return False
    else:
        print("⏭️  Skipping model loading tests.")
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! VitPose ROS wrapper is ready to use.")
    print("\nTo run the node:")
    print("1. Build the workspace: colcon build --packages-select mp_wrapper_ros")
    print("2. Source the workspace: source install/setup.bash")
    print("3. Launch the node: ros2 launch mp_wrapper_ros vitpose_ros_wrapper.launch.py")
    print("4. Or run directly: ros2 run mp_wrapper_ros vitpose_ros_wrapper")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()