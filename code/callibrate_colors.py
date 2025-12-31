import cv2
import numpy as np
import json
from pathlib import Path
import argparse

class ColorCalibrator:
    """Interactive tool for calibrating ball color detection"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Default HSV ranges for different colors
        self.color_presets = {
            'red': {'h_min1': 0, 'h_max1': 10, 'h_min2': 160, 'h_max2': 180, 
                   's_min': 100, 's_max': 255, 'v_min': 100, 'v_max': 255},
            'white': {'h_min1': 0, 'h_max1': 180, 'h_min2': 0, 'h_max2': 0,
                     's_min': 0, 's_max': 30, 'v_min': 200, 'v_max': 255},
            'pink': {'h_min1': 140, 'h_max1': 170, 'h_min2': 0, 'h_max2': 0,
                    's_min': 50, 's_max': 255, 'v_min': 100, 'v_max': 255},
            'orange': {'h_min1': 10, 'h_max1': 25, 'h_min2': 0, 'h_max2': 0,
                      's_min': 100, 's_max': 255, 'v_min': 100, 'v_max': 255},
            'yellow': {'h_min1': 20, 'h_max1': 35, 'h_min2': 0, 'h_max2': 0,
                      's_min': 100, 's_max': 255, 'v_min': 100, 'v_max': 255}
        }
        
        self.current_params = self.color_presets['red'].copy()
        self.current_frame = None
        self.frame_idx = 0
    
    def nothing(self, x):
        """Dummy callback for trackbars"""
        pass
    
    def load_preset(self, color_name):
        """Load preset color values"""
        if color_name in self.color_presets:
            self.current_params = self.color_presets[color_name].copy()
            print(f"Loaded preset: {color_name.upper()}")
            return True
        return False
    
    def update_trackbars(self, window_name):
        """Update trackbar positions"""
        cv2.setTrackbarPos('H min1', window_name, self.current_params['h_min1'])
        cv2.setTrackbarPos('H max1', window_name, self.current_params['h_max1'])
        cv2.setTrackbarPos('H min2', window_name, self.current_params['h_min2'])
        cv2.setTrackbarPos('H max2', window_name, self.current_params['h_max2'])
        cv2.setTrackbarPos('S min', window_name, self.current_params['s_min'])
        cv2.setTrackbarPos('S max', window_name, self.current_params['s_max'])
        cv2.setTrackbarPos('V min', window_name, self.current_params['v_min'])
        cv2.setTrackbarPos('V max', window_name, self.current_params['v_max'])
    
    def apply_color_detection(self, frame):
        """Apply color detection with current parameters"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks
        lower1 = np.array([self.current_params['h_min1'], 
                          self.current_params['s_min'], 
                          self.current_params['v_min']])
        upper1 = np.array([self.current_params['h_max1'], 
                          self.current_params['s_max'], 
                          self.current_params['v_max']])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Second range (for red which wraps around at 180)
        if self.current_params['h_min2'] > 0 or self.current_params['h_max2'] > 0:
            lower2 = np.array([self.current_params['h_min2'], 
                              self.current_params['s_min'], 
                              self.current_params['v_min']])
            upper2 = np.array([self.current_params['h_max2'], 
                              self.current_params['s_max'], 
                              self.current_params['v_max']])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = mask1
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and find best match
        result = frame.copy()
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 5000:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            score = circularity * np.sqrt(area) / 100
            
            if circularity > 0.5:
                # Draw all candidates in yellow
                cv2.drawContours(result, [contour], -1, (0, 255, 255), 2)
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        # Draw best match in green
        if best_contour is not None:
            cv2.drawContours(result, [best_contour], -1, (0, 255, 0), 3)
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(result, (cx, cy), 10, (0, 255, 0), -1)
                
                # Show detection info
                area = cv2.contourArea(best_contour)
                perimeter = cv2.arcLength(best_contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                info = f"Area: {int(area)} Circ: {circularity:.2f}"
                cv2.putText(result, info, (cx + 15, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result, mask
    
    def run_interactive(self, start_frame=0):
        """Run interactive calibration tool"""
        
        # Jump to start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.frame_idx = start_frame
        
        # Create windows
        window_name = 'Color Calibration'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 700)
        
        # Create trackbars
        cv2.createTrackbar('H min1', window_name, self.current_params['h_min1'], 180, self.nothing)
        cv2.createTrackbar('H max1', window_name, self.current_params['h_max1'], 180, self.nothing)
        cv2.createTrackbar('H min2', window_name, self.current_params['h_min2'], 180, self.nothing)
        cv2.createTrackbar('H max2', window_name, self.current_params['h_max2'], 180, self.nothing)
        cv2.createTrackbar('S min', window_name, self.current_params['s_min'], 255, self.nothing)
        cv2.createTrackbar('S max', window_name, self.current_params['s_max'], 255, self.nothing)
        cv2.createTrackbar('V min', window_name, self.current_params['v_min'], 255, self.nothing)
        cv2.createTrackbar('V max', window_name, self.current_params['v_max'], 255, self.nothing)
        
        print("\n" + "="*60)
        print("COLOR CALIBRATION TOOL")
        print("="*60)
        print("\nControls:")
        print("  r - Load RED preset")
        print("  w - Load WHITE preset")
        print("  p - Load PINK preset")
        print("  o - Load ORANGE preset")
        print("  y - Load YELLOW preset")
        print("  n - Next frame")
        print("  b - Previous frame")
        print("  s - Save parameters")
        print("  q - Quit")
        print("\nAdjust trackbars to isolate the ball (shown in GREEN)")
        print("="*60 + "\n")
        
        while True:
            # Read current frame if not cached
            if self.current_frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video or cannot read frame")
                    break
                self.current_frame = frame.copy()
            else:
                frame = self.current_frame.copy()
            
            # Get trackbar positions
            self.current_params['h_min1'] = cv2.getTrackbarPos('H min1', window_name)
            self.current_params['h_max1'] = cv2.getTrackbarPos('H max1', window_name)
            self.current_params['h_min2'] = cv2.getTrackbarPos('H min2', window_name)
            self.current_params['h_max2'] = cv2.getTrackbarPos('H max2', window_name)
            self.current_params['s_min'] = cv2.getTrackbarPos('S min', window_name)
            self.current_params['s_max'] = cv2.getTrackbarPos('S max', window_name)
            self.current_params['v_min'] = cv2.getTrackbarPos('V min', window_name)
            self.current_params['v_max'] = cv2.getTrackbarPos('V max', window_name)
            
            # Apply detection
            result, mask = self.apply_color_detection(frame)
            
            # Create display with original, result, and mask
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Resize for display
            h, w = frame.shape[:2]
            display_h = 600
            display_w = int(w * display_h / h)
            
            original_resized = cv2.resize(frame, (display_w, display_h))
            result_resized = cv2.resize(result, (display_w, display_h))
            mask_resized = cv2.resize(mask_colored, (display_w, display_h))
            
            # Add labels
            cv2.putText(original_resized, f'Original (Frame: {self.frame_idx})', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(result_resized, 'Detection Result', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(mask_resized, 'Color Mask', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Stack horizontally
            if display_w * 3 <= 1800:
                display = np.hstack([original_resized, result_resized, mask_resized])
            else:
                # Stack vertically if too wide
                display = np.vstack([
                    np.hstack([original_resized, result_resized]),
                    np.hstack([mask_resized, np.zeros_like(mask_resized)])
                ])
            
            cv2.imshow(window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                self.load_preset('red')
                self.update_trackbars(window_name)
            elif key == ord('w'):
                self.load_preset('white')
                self.update_trackbars(window_name)
            elif key == ord('p'):
                self.load_preset('pink')
                self.update_trackbars(window_name)
            elif key == ord('o'):
                self.load_preset('orange')
                self.update_trackbars(window_name)
            elif key == ord('y'):
                self.load_preset('yellow')
                self.update_trackbars(window_name)
            elif key == ord('n'):
                # Next frame
                self.current_frame = None
                self.frame_idx += 1
            elif key == ord('b'):
                # Previous frame
                if self.frame_idx > 0:
                    self.frame_idx -= 1
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
                    self.current_frame = None
            elif key == ord('s'):
                self.save_parameters()
        
        cv2.destroyAllWindows()
        self.cap.release()
    
    def save_parameters(self):
        """Save current parameters to file"""
        output_path = Path('color_calibration.json')
        
        with open(output_path, 'w') as f:
            json.dump(self.current_params, f, indent=2)
        
        print(f"\n✓ Parameters saved to: {output_path}")
        print("\nCurrent values:")
        for key, value in self.current_params.items():
            print(f"  {key}: {value}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description='Interactive tool for calibrating ball color detection'
    )
    parser.add_argument('--video', '-v', required=True,
                       help='Path to cricket video')
    parser.add_argument('--frame', '-f', type=int, default=0,
                       help='Starting frame number (default: 0)')
    
    args = parser.parse_args()
    
    calibrator = ColorCalibrator(args.video)
    calibrator.run_interactive(args.frame)

if __name__ == "__main__":
    main()