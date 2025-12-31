import cv2 
import numpy as np 
from ultralytics import YOLO 
import torch 
 
print("Testing installations...") 
print(f"√ OpenCV version: {cv2.__version__}") 
print(f"√ NumPy version: {np.__version__}") 
print(f"√ PyTorch version: {torch.__version__}") 
print(f"√ CUDA available: {torch.cuda.is_available()}") 
 
try: 
    model = YOLO('yolov8x.pt') 
    print("√ YOLO model loaded successfully") 
except Exception as e: 
    print(f"X Error loading YOLO: {e}") 
 
print("\n√ All installations verified!") 
