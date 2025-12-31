from pathlib import Path
from cricket_ball_tracker import CricketBallTracker
import argparse
import json

# Global output dirs (tracker contract)
RESULTS_DIR = Path("results")
ANNOTATIONS_DIR = Path("annotations")

RESULTS_DIR.mkdir(exist_ok=True)
ANNOTATIONS_DIR.mkdir(exist_ok=True)


def process_all_videos(input_dir, summary_dir='results', ball_color='auto'):
    input_path = Path(input_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f'*{ext}'))

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} video(s) to process")
    print(f"Ball color mode: {ball_color}")
    print("=" * 60)

    all_results = []

    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 60)

        try:
            tracker = CricketBallTracker(
                str(video_path),
                ball_color=ball_color
            )

            annotations = tracker.process_video()

            # ---------------- VERIFY OUTPUT VIDEO ---------------- #
            stem = video_path.stem

            mp4_path = RESULTS_DIR / f"{stem}.mp4"
            avi_path = RESULTS_DIR / f"{stem}.avi"


            if mp4_path.exists():
                output_video = mp4_path
            elif avi_path.exists():
                output_video = avi_path
            else:
                raise RuntimeError(
                    f"Tracked video NOT saved (.mp4 or .avi missing) for {video_path.name}"
                )

            # ---------------- STATS ---------------- #
            visible_frames = sum(a['visible'] for a in annotations)
            total_frames = len(annotations)
            detection_rate = (visible_frames / total_frames) * 100

            result = {
                        'video_name': video_path.name,
                        'output_video': output_video.name,
                        'total_frames': total_frames,
                        'visible_frames': visible_frames,
                        'detection_rate': detection_rate
                    }

            all_results.append(result)

            print(f"Tracked video saved at: {output_video}")
            print(f"Annotations saved at: annotations/{video_path.stem}.csv")
            print(f"Detection rate: {detection_rate:.1f}%")

        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ---------------- SUMMARY ---------------- #
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)

    if not all_results:
        print("No videos processed successfully.")
        return

    avg_detection = sum(r['detection_rate'] for r in all_results) / len(all_results)

    summary = {
        'total_videos': len(all_results),
        'average_detection_rate': avg_detection,
        'videos': all_results
    }

    summary_dir = Path(summary_dir)
    summary_dir.mkdir(exist_ok=True)
    summary_path = summary_dir / "processing_summary.json"

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch cricket ball tracking")

    parser.add_argument('--input', '-i', required=True,
                        help='Input directory containing videos')
    parser.add_argument('--output', '-o', default='results',
                        help='Directory for summary JSON')
    parser.add_argument('--color', '-c', default='auto',
                        choices=['auto', 'red', 'white', 'pink', 'orange', 'yellow'],
                        help='Ball color')

    args = parser.parse_args()
    process_all_videos(args.input, args.output, args.color)
