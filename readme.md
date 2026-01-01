# Cricket Ball Tracking System
**EdgeFleet.AI Assessment - IIT BHU**

A computer vision system for detecting and tracking cricket balls in videos from fixed cameras using a hybrid approach combining deep learning (YOLOv8) and classical computer vision techniques.

---

## 🎯 Features

- **Multi-Color Ball Support**: Automatically detects RED, WHITE, PINK, ORANGE, and YELLOW cricket balls
- **Multi-Method Detection**: Combines YOLOv8 (pre-trained) with color-based detection for robustness
- **Kalman Filter Tracking**: Smooth trajectory estimation and occlusion handling
- **Zero Training Required**: Uses pre-trained models on COCO dataset
- **Interactive Color Calibration**: Fine-tune color detection for specific lighting conditions
- **High Accuracy**: Optimized for stationary camera scenarios
- **Complete Pipeline**: Detection → Tracking → Visualization → Evaluation

---

## 📁 Repository Structure

```
cricket-ball-tracker/
│
├── code/
│   ├── cricket_ball_tracker.py    # Main tracking class
│   ├── batch_process.py            # Batch processing script
│   ├── evaluate.py                 # Evaluation and visualization
│   └── calibrate_colors.py         # Interactive color calibration tool
│
├── annotations/                    # Output CSV/JSON annotations
│
├── results/
│   ├── *_tracked.mp4              # Processed videos with overlays
│   └── evaluation/                # Evaluation plots and reports
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── report.pdf                      # Detailed technical report
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd cricket-ball-tracker

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Test Dataset

Download the test videos from:
```
https://drive.google.com/file/d/1hnaGuqGuMXaFKI5fhfy8gatzCH-6iMcJ/view?usp=sharing
```

Extract to a folder (e.g., `test_videos/`)

### 3. Process Videos

**Single video (auto-detect color):**
```bash
python code/cricket_ball_tracker.py
# Edit the video path in the script's main() function
```

**Batch processing (auto-detect colors):**
```bash
python code/batch_process.py --input test_videos/ --output results/
```

**Specify ball color if known:**
```bash
# For white ball cricket
python code/batch_process.py --input test_videos/ --output results/ --color white

# For pink ball cricket (day/night)
python code/batch_process.py --input test_videos/ --output results/ --color pink
```

**Available color options:** `auto`, `red`, `white`, `pink`, `orange`, `yellow`

### 4. Evaluate Results

```bash
python code/evaluate.py --annotations results/annotations/video_name_annotations.csv --output results/evaluation/
```

### 5. (Optional) Calibrate Colors

If automatic color detection doesn't work well for your videos:

```bash
# Launch interactive calibration tool
python code/calibrate_colors.py --video test_videos/your_video.mp4 --frame 100

# Controls:
# r - Load RED preset
# w - Load WHITE preset  
# p - Load PINK preset
# o - Load ORANGE preset
# n/b - Next/Previous frame
# s - Save parameters
# q - Quit
```

The tool will save calibrated parameters to `color_calibration.json`.

---

## 📊 Output Format

### Annotation Files (CSV)

```csv
frame,x,y,visible
0,512.3,298.1,1
1,518.7,305.4,1
2,-1,-1,0
```

- **frame**: Frame index (0-based)
- **x, y**: Ball centroid coordinates (pixels)
- **visible**: 1 if ball detected, 0 if not visible

### Video Output

- MP4 video with:
  - Green circle at ball centroid
  - Blue trajectory line
  - "Ball" label

---

## 🧠 Technical Approach

### 1. Detection Stage

**Primary Method: YOLOv8**
- Uses pre-trained YOLOv8x model (largest, most accurate)
- Trained on COCO dataset which includes "sports ball" class
- No additional training required
- Confidence threshold: 0.3

**Fallback Method: Color Segmentation**
- Detects red cricket balls using HSV color space
- Circular contour detection
- Activated when YOLO fails

### 2. Tracking Stage

**Kalman Filter Implementation**
- 4D state space: [x, y, vx, vy]
- Constant velocity motion model
- Handles temporary occlusions
- Smooths noisy detections

### 3. Post-Processing

- Trajectory smoothing
- Occlusion interpolation
- Outlier rejection

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| YOLO Model | yolov8x.pt | Largest YOLOv8 variant |
| Confidence Threshold | 0.3 | Minimum detection confidence |
| Ball Class ID | 32 | Sports ball in COCO |
| Kalman Q (Process Noise) | 0.1 | Motion uncertainty |
| Kalman R (Measurement Noise) | 10 | Measurement uncertainty |
| **Color Ranges (HSV)** | | **Auto-detected** |
| Red Ball | H:[0-10, 160-180] S:[100-255] V:[100-255] | Standard red cricket ball |
| White Ball | H:[0-180] S:[0-30] V:[200-255] | White ball cricket |
| Pink Ball | H:[140-170] S:[50-255] V:[100-255] | Day/night cricket |
| Orange Ball | H:[10-25] S:[100-255] V:[100-255] | Training/indoor cricket |
| Yellow Ball | H:[20-35] S:[100-255] V:[100-255] | Tennis ball cricket |

---

## 🔍 Key Design Decisions

### Why YOLOv8?
- Pre-trained on diverse sports scenarios
- No training data required
- High accuracy on small objects
- Real-time capable (though not needed here)

### Why Kalman Filter?
- Stationary camera = predictable motion
- Handles occlusions gracefully
- Smooths jittery detections
- Computationally efficient

### Why Hybrid Approach?
- YOLO may fail in poor lighting/fast motion
- Different videos use different colored balls (red/white/pink)
- Color detection provides robust fallback
- Automatic color detection adapts to each video
- Combines strengths of both methods

### Handling Challenges

**Challenge 1: Ball Occlusion**
- Solution: Kalman filter predicts position during occlusion
- Interpolation for short gaps

**Challenge 2: Motion Blur**
- Solution: Color detection + morphological operations
- Larger confidence thresholds for YOLO

**Challenge 3: Similar Objects**
- Solution: Circularity check for color detection
- Temporal consistency (trajectory validation)

---

## 📈 Performance Metrics

The evaluation script generates:

1. **Detection Rate**: % of frames where ball is visible
2. **Trajectory Plots**: 2D spatial trajectory with time color-coding
3. **Position Analysis**: X/Y coordinates over time
4. **Velocity Profile**: Frame-to-frame movement speed
5. **Detection Timeline**: Visual representation of tracking continuity

---

## 🛠️ Troubleshooting

**Issue: Low detection rate**
- Adjust `min_confidence` threshold (try 0.2-0.5)
- Run color calibration tool: `python code/calibrate_colors.py --video your_video.mp4`
- Try specifying ball color manually: `--color white` or `--color pink`
- Check video quality and lighting

**Issue: Wrong ball color detected**
- Use color calibration tool to fine-tune
- Specify color manually: `python code/batch_process.py -i videos/ -c white`
- Check if multiple colored objects are in frame

**Issue: Ball changes color mid-video**
- Current system detects one color per video
- Process video segments separately if ball color changes

**Issue: Jittery tracking**
- Increase Kalman Q noise (makes predictions smoother)
- Decrease R noise (trusts measurements more)

**Issue: False positives**
- Increase confidence threshold
- Add size constraints in color detection
- Implement temporal consistency checks

---

## 📝 Dependencies

- **OpenCV**: Image processing and video I/O
- **NumPy**: Numerical computations
- **Pandas**: Data handling
- **Ultralytics**: YOLOv8 implementation
- **FilterPy**: Kalman filter
- **PyTorch**: Deep learning backend
- **Matplotlib/Seaborn**: Visualization

---

## 🎓 Future Improvements

1. **Multi-Ball Tracking**: Extend to track multiple balls simultaneously
2. **3D Trajectory Estimation**: Camera calibration + physics modeling
3. **Event Detection**: Detect ball bounces, catches, boundaries
4. **Player Tracking**: Integrate player detection for context
5. **Real-time Processing**: Optimize for live streaming scenarios
