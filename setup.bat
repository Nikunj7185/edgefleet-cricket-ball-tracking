@echo off
echo ==========================================
echo Cricket Ball Tracker - Setup Script
echo ==========================================
echo.

REM Check Python
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)
python --version
echo √ Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo √ Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo √ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo √ pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo √ All dependencies installed successfully
) else (
    echo X Error installing dependencies
    pause
    exit /b 1
)
echo.

REM Create directory structure
echo Creating directory structure...
if not exist code mkdir code
if not exist annotations mkdir annotations
if not exist results mkdir results
if not exist results\evaluation mkdir results\evaluation
if not exist test_videos mkdir test_videos
echo √ Directories created
echo.

REM Download YOLO model
echo Downloading YOLOv8 model...
python -c "from ultralytics import YOLO; YOLO('yolov8x.pt')"
if %errorlevel% equ 0 (
    echo √ YOLOv8 model downloaded
) else (
    echo X Error downloading model
)
echo.

REM Create test script
echo import cv2 > test_installation.py
echo import numpy as np >> test_installation.py
echo from ultralytics import YOLO >> test_installation.py
echo import torch >> test_installation.py
echo. >> test_installation.py
echo print("Testing installations...") >> test_installation.py
echo print(f"√ OpenCV version: {cv2.__version__}") >> test_installation.py
echo print(f"√ NumPy version: {np.__version__}") >> test_installation.py
echo print(f"√ PyTorch version: {torch.__version__}") >> test_installation.py
echo print(f"√ CUDA available: {torch.cuda.is_available()}") >> test_installation.py
echo. >> test_installation.py
echo try: >> test_installation.py
echo     model = YOLO('yolov8x.pt') >> test_installation.py
echo     print("√ YOLO model loaded successfully") >> test_installation.py
echo except Exception as e: >> test_installation.py
echo     print(f"X Error loading YOLO: {e}") >> test_installation.py
echo. >> test_installation.py
echo print("\n√ All installations verified!") >> test_installation.py

echo Testing installation...
python test_installation.py
echo.

REM Print next steps
echo ==========================================
echo Setup Complete! 🎉
echo ==========================================
echo.
echo Next steps:
echo 1. Download test videos from the Google Drive link
echo 2. Place videos in the 'test_videos\' directory
echo 3. Run: python code\batch_process.py --input test_videos\ --output results\
echo.
echo To activate the environment later, run:
echo   venv\Scripts\activate.bat
echo.
echo For more information, see README.md
echo ==========================================
pause