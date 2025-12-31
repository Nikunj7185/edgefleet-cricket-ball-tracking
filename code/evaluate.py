import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import argparse

class TrackingEvaluator:
    def __init__(self, annotations_path):
        self.annotations_path = Path(annotations_path)
        self.df = pd.read_csv(annotations_path)
        
    def generate_report(self, output_dir='results/evaluation'):
        """Generate comprehensive evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("TRACKING EVALUATION REPORT")
        print("="*60)
        
        # Basic statistics
        total_frames = len(self.df)
        visible_frames = self.df['visible'].sum()
        invisible_frames = total_frames - visible_frames
        detection_rate = (visible_frames / total_frames) * 100
        
        print(f"\n1. DETECTION STATISTICS")
        print("-"*60)
        print(f"Total frames processed: {total_frames}")
        print(f"Ball visible: {visible_frames} frames ({detection_rate:.2f}%)")
        print(f"Ball not visible: {invisible_frames} frames ({100-detection_rate:.2f}%)")
        
        # Trajectory analysis
        visible_data = self.df[self.df['visible'] == 1]
        
        if len(visible_data) > 0:
            print(f"\n2. TRAJECTORY ANALYSIS")
            print("-"*60)
            print(f"X position range: [{visible_data['x'].min():.1f}, {visible_data['x'].max():.1f}]")
            print(f"Y position range: [{visible_data['y'].min():.1f}, {visible_data['y'].max():.1f}]")
            print(f"Mean X position: {visible_data['x'].mean():.1f}")
            print(f"Mean Y position: {visible_data['y'].mean():.1f}")
            
            # Calculate velocity (frame-to-frame movement)
            visible_data = visible_data.copy()
            visible_data['dx'] = visible_data['x'].diff()
            visible_data['dy'] = visible_data['y'].diff()
            visible_data['velocity'] = np.sqrt(visible_data['dx']**2 + visible_data['dy']**2)
            
            mean_velocity = visible_data['velocity'].mean()
            max_velocity = visible_data['velocity'].max()
            
            print(f"\n3. MOTION ANALYSIS")
            print("-"*60)
            print(f"Mean velocity (pixels/frame): {mean_velocity:.2f}")
            print(f"Max velocity (pixels/frame): {max_velocity:.2f}")
            
            # Detect tracking gaps
            frame_diffs = visible_data['frame'].diff()
            gaps = frame_diffs[frame_diffs > 1]
            
            print(f"\n4. TRACKING CONTINUITY")
            print("-"*60)
            print(f"Number of tracking gaps: {len(gaps)}")
            if len(gaps) > 0:
                print(f"Longest gap: {gaps.max()-1:.0f} frames")
                print(f"Average gap length: {gaps.mean()-1:.2f} frames")
            
            # Generate visualizations
            self._plot_trajectory(visible_data, output_path)
            self._plot_position_over_time(visible_data, output_path)
            self._plot_velocity(visible_data, output_path)
            self._plot_detection_timeline(self.df, output_path)
            
            print(f"\n5. VISUALIZATIONS")
            print("-"*60)
            print(f"Plots saved to: {output_path}/")
            print("  - trajectory_2d.png")
            print("  - position_over_time.png")
            print("  - velocity_profile.png")
            print("  - detection_timeline.png")
        
        # Save report to text file
        report_path = output_path / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CRICKET BALL TRACKING EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Annotation file: {self.annotations_path}\n")
            f.write(f"Total frames: {total_frames}\n")
            f.write(f"Detection rate: {detection_rate:.2f}%\n")
            f.write(f"Visible frames: {visible_frames}\n")
            f.write(f"Invisible frames: {invisible_frames}\n")
            
            if len(visible_data) > 0:
                f.write(f"\nMean velocity: {mean_velocity:.2f} pixels/frame\n")
                f.write(f"Max velocity: {max_velocity:.2f} pixels/frame\n")
                f.write(f"Tracking gaps: {len(gaps)}\n")
        
        print(f"\nReport saved to: {report_path}")
        print("="*60)
    
    def _plot_trajectory(self, data, output_dir):
        """Plot 2D trajectory of the ball"""
        plt.figure(figsize=(12, 8))
        
        # Color code by time
        scatter = plt.scatter(data['x'], data['y'], 
                            c=data['frame'], 
                            cmap='viridis', 
                            s=10, 
                            alpha=0.6)
        
        # Draw trajectory line
        plt.plot(data['x'], data['y'], 'r-', alpha=0.3, linewidth=1)
        
        plt.colorbar(scatter, label='Frame Number')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Cricket Ball Trajectory (2D View)')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_position_over_time(self, data, output_dir):
        """Plot X and Y positions over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        ax1.plot(data['frame'], data['x'], 'b-', linewidth=1.5)
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('X Position (pixels)')
        ax1.set_title('Horizontal Position Over Time')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(data['frame'], data['y'], 'r-', linewidth=1.5)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.set_title('Vertical Position Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'position_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_velocity(self, data, output_dir):
        """Plot velocity profile"""
        plt.figure(figsize=(14, 6))
        
        plt.plot(data['frame'], data['velocity'], 'g-', linewidth=1.5, alpha=0.7)
        plt.fill_between(data['frame'], data['velocity'], alpha=0.3, color='green')
        
        plt.xlabel('Frame Number')
        plt.ylabel('Velocity (pixels/frame)')
        plt.title('Ball Velocity Profile')
        plt.grid(True, alpha=0.3)
        
        # Add mean line
        mean_vel = data['velocity'].mean()
        plt.axhline(y=mean_vel, color='r', linestyle='--', 
                   label=f'Mean: {mean_vel:.2f}', linewidth=2)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_profile.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detection_timeline(self, data, output_dir):
        """Plot detection timeline showing when ball was visible"""
        plt.figure(figsize=(14, 4))
        
        colors = ['red' if v == 0 else 'green' for v in data['visible']]
        plt.bar(data['frame'], data['visible'], 
               color=colors, width=1.0, alpha=0.6)
        
        plt.xlabel('Frame Number')
        plt.ylabel('Ball Detected')
        plt.title('Ball Detection Timeline (Green=Visible, Red=Not Visible)')
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate cricket ball tracking results')
    parser.add_argument('--annotations', '-a', required=True, 
                       help='Path to annotations CSV file')
    parser.add_argument('--output', '-o', default='results/evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    evaluator = TrackingEvaluator(args.annotations)
    evaluator.generate_report(args.output)

if __name__ == "__main__":
    main()