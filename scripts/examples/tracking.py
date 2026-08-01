import pickle
from argparse import ArgumentParser
from pathlib import Path

import cv2
import supervision as sv
from trackers import ByteTrackTracker
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        required=True,
        help='Path where predictions for video are stored'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input_dir / 'preictions.pkl', 'rb') as f:
        preds = pickle.load(f)

    tracker = ByteTrackTracker()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frames = []
    for detections in preds:
        detections = tracker.update(detections=detections)

        labels = [f"#{det[4]} {det[5]['class_name']}" for det in detections]

        if 'source_image' not in detections.metadata:
            continue

        annotated_frame = box_annotator.annotate(detections.metadata['source_image'].copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        frames.append(annotated_frame)

    # Save gif
    gif_path = args.input_dir / 'tracking.gif'
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 / 30,  # ms
        loop=0,
    )
    del pil_frames
    print(f'GIF saved to {gif_path}')

    # Save video
    video_path = args.input_dir / 'tracking.avi'
    fps = 30
    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    if not video.isOpened():
        raise RuntimeError('VideoWriter failed to open')

    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(bgr_frame)

    video.release()
    print(f'Video saved to {video_path}')


if __name__ == '__main__':
    main()
