import cv2
import numpy as np
from pathlib import Path
import json
from cricket_ball_tracker import CricketBallTracker
import matplotlib.pyplot as plt

class ParameterTuner:
    """
    Interactive tool for tuning detection and tracking parameters
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Default parameters
        self.params = {
            'yolo_confidence': 0.3,
            'color_lower_red1': [0, 100, 100],
            'color_upper_red1': [10, 255, 255],
            'color_lower_red2': [160, 100, 100],
            'color_upper_red2': [180, 255, 255],
            'min_area': 50,
            'max_area': 5000,
            'min_circularity': 0.5,
            'kalman_q': 0.1,
            'kalman_r': 10
        }
        
        self.results = []
    
    def test_frame_range(self, start_frame=0, num_frames=100):
        """
        Test current parameters on a range of frames
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        detections = []
        for i in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Test YOLO detection
            tracker = CricketBallTracker(self.video_path)
            tracker.min_confidence = self.params['yolo_confidence']
            yolo_det = tracker.detect_ball_yolo(frame)
            
            # Test color detection
            color_det = self._test_color_detection(frame)
            
            detections.append({
                'frame': start_frame + i,
                'yolo_detected': yolo_det is not None,
                'color_detected': color_det is not None
            })
        
        # Calculate metrics
        yolo_rate = sum(1 for d in detections if d['yolo_detected']) / len(detections)
        color_rate = sum(1 for d in detections if d['color_detected']) / len(detections)
        
        result = {
            'params': self.params.copy(),
            'yolo_detection_rate': yolo_rate,
            'color_detection_rate': color_rate,
            'total_detections': yolo_rate + color_rate  # Combined
        }
        
        self.results.append(result)
        
        print(f"\nTest Results (Frames {start_frame}-{start_frame+num_frames}):")
        print(f"  YOLO Detection Rate: {yolo_rate*100:.1f}%")
        print(f"  Color Detection Rate: {color_rate*100:.1f}%")
        print(f"  Combined: {(yolo_rate + color_rate)*100:.1f}%")
        
        return result
    
    def _test_color_detection(self, frame):
        """Test color detection with current parameters"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array(self.params['color_lower_red1'])
        upper_red1 = np.array(self.params['color_upper_red1'])
        lower_red2 = np.array(self.params['color_lower_red2'])
        upper_red2 = np.array(self.params['color_upper_red2'])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            best_contour = None
            best_circularity = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.params['min_area'] or area > self.params['max_area']:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                if circularity > best_circularity and circularity > self.params['min_circularity']:
                    best_circularity = circularity
                    best_contour = contour
            
            if best_contour is not None:
                M = cv2.moments(best_contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    return (cx, cy, best_circularity)
        
        return None
    
    def grid_search_yolo_confidence(self, start_frame=0, num_frames=100):
        """Grid search for optimal YOLO confidence threshold"""
        print("\n" + "="*60)
        print("GRID SEARCH: YOLO Confidence Threshold")
        print("="*60)
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        best_result = None
        best_score = 0
        
        for conf in thresholds:
            self.params['yolo_confidence'] = conf
            result = self.test_frame_range(start_frame, num_frames)
            
            if result['yolo_detection_rate'] > best_score:
                best_score = result['yolo_detection_rate']
                best_result = result
        
        print(f"\n✓ Best YOLO Confidence: {best_result['params']['yolo_confidence']}")
        print(f"  Detection Rate: {best_score*100:.1f}%")
        
        self.params['yolo_confidence'] = best_result['params']['yolo_confidence']
        return best_result
    
    def tune_color_range(self, start_frame=0, num_frames=50):
        """Interactive color range tuning"""
        print("\n" + "="*60)
        print("COLOR RANGE TUNING")
        print("="*60)
        print("Adjust HSV ranges for red ball detection")
        print("Press 'q' to quit, 's' to save current settings")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = self.cap.read()
        
        if not ret:
            print("Error reading frame")
            return
        
        def nothing(x):
            pass
        
        # Create window with trackbars
        cv2.namedWindow('Color Tuning')
        cv2.createTrackbar('H_min1', 'Color Tuning', 0, 180, nothing)
        cv2.createTrackbar('S_min', 'Color Tuning', 100, 255, nothing)
        cv2.createTrackbar('V_min', 'Color Tuning', 100, 255, nothing)
        cv2.createTrackbar('H_max1', 'Color Tuning', 10, 180, nothing)
        cv2.createTrackbar('H_min2', 'Color Tuning', 160, 180, nothing)
        cv2.createTrackbar('H_max2', 'Color Tuning', 180, 180, nothing)
        
        while True:
            h_min1 = cv2.getTrackbarPos('H_min1', 'Color Tuning')
            s_min = cv2.getTrackbarPos('S_min', 'Color Tuning')
            v_min = cv2.getTrackbarPos('V_min', 'Color Tuning')
            h_max1 = cv2.getTrackbarPos('H_max1', 'Color Tuning')
            h_min2 = cv2.getTrackbarPos('H_min2', 'Color Tuning')
            h_max2 = cv2.getTrackbarPos('H_max2', 'Color Tuning')
            
            # Apply color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower_red1 = np.array([h_min1, s_min, v_min])
            upper_red1 = np.array([h_max1, 255, 255])
            lower_red2 = np.array([h_min2, s_min, v_min])
            upper_red2 = np.array([h_max2, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Show results
            result = cv2.bitwise_and(frame, frame, mask=mask)
            combined = np.hstack([frame, result])
            
            cv2.imshow('Color Tuning', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.params['color_lower_red1'] = [h_min1, s_min, v_min]
                self.params['color_upper_red1'] = [h_max1, 255, 255]
                self.params['color_lower_red2'] = [h_min2, s_min, v_min]
                self.params['color_upper_red2'] = [h_max2, 255, 255]
                print("\n✓ Color ranges saved!")
                print(f"  Range 1: {self.params['color_lower_red1']} - {self.params['color_upper_red1']}")
                print(f"  Range 2: {self.params['color_lower_red2']} - {self.params['color_upper_red2']}")
        
        cv2.destroyAllWindows()
    
    def save_best_params(self, output_path='results/best_params.json'):
        """Save best parameters to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.params, f, indent=2)
        
        print(f"\n✓ Best parameters saved to: {output_path}")
    
    def plot_tuning_results(self, output_path='results/tuning_results.png'):
        """Plot parameter tuning results"""
        if not self.results:
            print("No tuning results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # YOLO confidence vs detection rate
        yolo_confs = [r['params']['yolo_confidence'] for r in self.results]
        yolo_rates = [r['yolo_detection_rate'] * 100 for r in self.results]
        
        ax1.plot(yolo_confs, yolo_rates, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('YOLO Confidence Threshold')
        ax1.set_ylabel('Detection Rate (%)')
        ax1.set_title('YOLO Confidence Tuning')
        ax1.grid(True, alpha=0.3)
        
        # Combined detection rates
        combined_rates = [r['total_detections'] * 100 for r in self.results]
        
        ax2.bar(range(len(self.results)), combined_rates, color='green', alpha=0.6)
        ax2.set_xlabel('Test Run')
        ax2.set_ylabel('Combined Detection Rate (%)')
        ax2.set_title('Overall Detection Performance')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Tuning results plot saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    tuner = ParameterTuner('path/to/cricket_video.mp4')
    
    # Test current parameters
    tuner.test_frame_range(start_frame=0, num_frames=100)
    
    # Grid search for best YOLO confidence
    tuner.grid_search_yolo_confidence(start_frame=0, num_frames=100)
    
    # Interactive color tuning (optional)
    # tuner.tune_color_range(start_frame=50)
    
    # Save best parameters
    tuner.save_best_params()
    
    # Plot results
    tuner.plot_tuning_results()