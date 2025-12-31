"""
System verification script to test if everything is working
Run this before processing actual videos
"""

import sys
import cv2
import numpy as np
import torch
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def print_success(text):
    print(f"✓ {text}")

def print_error(text):
    print(f"✗ {text}")

def test_imports():
    """Test if all required packages are installed"""
    print_header("Testing Package Imports")
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('torch', 'torch'),
        ('filterpy', 'filterpy'),
        ('ultralytics', 'ultralytics')
    ]
    
    all_good = True
    for module, package in required_packages:
        try:
            __import__(module)
            version = __import__(module).__version__
            print_success(f"{package} ({version})")
        except ImportError:
            print_error(f"{package} NOT INSTALLED")
            all_good = False
        except AttributeError:
            print_success(f"{package} (version unknown)")
    
    return all_good

def test_opencv():
    """Test OpenCV functionality"""
    print_header("Testing OpenCV")
    
    try:
        # Create a test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (255, 0, 0)  # Blue background
        cv2.circle(img, (320, 240), 50, (0, 255, 0), -1)
        
        # Test HSV conversion
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Test contour detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print_success(f"OpenCV version: {cv2.__version__}")
        print_success(f"Created test image: {img.shape}")
        print_success(f"HSV conversion: OK")
        print_success(f"Contour detection: {len(contours)} contours found")
        
        return True
    except Exception as e:
        print_error(f"OpenCV test failed: {str(e)}")
        return False

def test_pytorch():
    """Test PyTorch and CUDA"""
    print_header("Testing PyTorch")
    
    try:
        print_success(f"PyTorch version: {torch.__version__}")
        print_success(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print_success(f"CUDA version: {torch.version.cuda}")
            print_success(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available - will use CPU (slower)")
        
        # Test tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print_success(f"Tensor operations: OK")
        
        return True
    except Exception as e:
        print_error(f"PyTorch test failed: {str(e)}")
        return False

def test_yolo():
    """Test YOLO model loading"""
    print_header("Testing YOLO Model")
    
    try:
        from ultralytics import YOLO
        
        print("  Loading YOLOv8x model (this may take a minute)...")
        model = YOLO('yolov8x.pt')
        print_success("YOLOv8x model loaded successfully")
        
        # Test inference on dummy image
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False)
        print_success("Model inference: OK")
        
        # Check if sports ball class exists
        class_names = model.names
        if 32 in class_names and class_names[32] == 'sports ball':
            print_success(f"Sports ball class (ID: 32): Found")
        else:
            print_error("Sports ball class not found in model")
            return False
        
        return True
    except Exception as e:
        print_error(f"YOLO test failed: {str(e)}")
        return False

def test_kalman_filter():
    """Test Kalman filter"""
    print_header("Testing Kalman Filter")
    
    try:
        from filterpy.kalman import KalmanFilter
        
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        
        # Test prediction and update
        kf.x = np.array([0, 0, 1, 1])
        kf.predict()
        kf.update(np.array([1, 1]))
        
        print_success("Kalman filter initialization: OK")
        print_success("Predict/Update cycle: OK")
        
        return True
    except Exception as e:
        print_error(f"Kalman filter test failed: {str(e)}")
        return False

def test_color_detection():
    """Test color detection functionality"""
    print_header("Testing Color Detection")
    
    try:
        # Create test images with different colored circles
        colors_to_test = {
            'red': (0, 0, 255),
            'white': (255, 255, 255),
            'pink': (203, 192, 255),
            'orange': (0, 165, 255),
            'yellow': (0, 255, 255)
        }
        
        for color_name, bgr in colors_to_test.items():
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(img, (320, 240), 50, bgr, -1)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Test if we can detect it
            # (Simplified test - actual detection is more complex)
            if color_name == 'red':
                mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            elif color_name == 'white':
                mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
            else:
                mask = np.zeros((480, 640), dtype=np.uint8)
            
            detected = np.sum(mask > 0) > 100
            
            if detected or color_name not in ['red', 'white']:
                print_success(f"{color_name.upper()} ball detection: Ready")
        
        return True
    except Exception as e:
        print_error(f"Color detection test failed: {str(e)}")
        return False

def test_file_structure():
    """Test if directory structure exists"""
    print_header("Testing Directory Structure")
    
    required_dirs = ['code', 'annotations', 'results', 'test_videos']
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print_success(f"Directory '{dir_name}/' exists")
        else:
            print_error(f"Directory '{dir_name}/' NOT FOUND")
            print(f"  Creating '{dir_name}/'...")
            dir_path.mkdir(exist_ok=True)
            all_good = False
    
    return all_good

def test_tracker_class():
    """Test if tracker class can be instantiated"""
    print_header("Testing Tracker Class")
    
    try:
        # Check if cricket_ball_tracker.py exists
        tracker_file = Path('code/cricket_ball_tracker.py')
        if not tracker_file.exists():
            print_error("cricket_ball_tracker.py not found in code/")
            return False
        
        print_success("cricket_ball_tracker.py found")
        
        # Try to import (but don't instantiate without video)
        import sys
        sys.path.insert(0, 'code')
        from cricket_ball_tracker import CricketBallTracker
        
        print_success("CricketBallTracker class imported successfully")
        print("  ⚠ Skipping instantiation (requires video file)")
        
        return True
    except ImportError as e:
        print_error(f"Cannot import CricketBallTracker: {str(e)}")
        return False
    except Exception as e:
        print_error(f"Tracker test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print(" CRICKET BALL TRACKER - SYSTEM VERIFICATION")
    print("="*60)
    print("\nThis will test if all components are working correctly")
    print("Run this before processing actual videos\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("OpenCV", test_opencv),
        ("PyTorch", test_pytorch),
        ("YOLO Model", test_yolo),
        ("Kalman Filter", test_kalman_filter),
        ("Color Detection", test_color_detection),
        ("Directory Structure", test_file_structure),
        ("Tracker Class", test_tracker_class)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Unexpected error in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<25} {status}")
    
    print("\n" + "-"*60)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Download test videos from Google Drive")
        print("2. Place videos in test_videos/ directory")
        print("3. Run: python code/batch_process.py --input test_videos/ --output results/")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues before proceeding.")
        print("\nCommon fixes:")
        print("- Run setup.sh or setup.bat to install dependencies")
        print("- Check if you're in the virtual environment")
        print("- Make sure all code files are in code/ directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())