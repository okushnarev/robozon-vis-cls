import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import h5py
import numpy as np
import supervision as sv
from tqdm import tqdm

# Add project root to PATH
if (project_root := str(Path.cwd())) not in sys.path:
    sys.path.append(project_root)

from src.classifiers import RFDETRClassifier
from src.measure.measurers import ClassicMeasurer
from src.wrappers import load_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--ckpt-dir',
        type=Path,
        required=True,
        help='Path to dir with model\'s checkpoint and config'
    )
    parser.add_argument(
        '--ckpt-type',
        type=str,
        default='last.ckpt',
        help='Checkpoint type'
    )
    parser.add_argument(
        '--intrinsics',
        type=Path,
        default=None,
        help='Path to the camera intrinsics file\n'
             'Expected a .json file with fx, fy, cx, cy, s keys\n')
    parser.add_argument(
        '--ds-dir',
        type=Path,
        required=True,
        help='Path where datasets are stored'
    )
    parser.add_argument(
        '--video-dir',
        type=Path,
        default=Path('videos'),
        help='Path where to output videos. Point to root video dir. Children directories are auto resolved'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    exp_name = args.ds_dir.name
    video_dir = args.video_dir / args.ds_dir.parent.name / exp_name
    video_dir.mkdir(exist_ok=True, parents=True)

    if args.intrinsics:
        with open(args.intrinsics, 'r') as intrinsics_file:
            intrinsics = json.load(intrinsics_file)
    else:
        # Fall back to default
        intrinsics = {
            "fx": 710.0132366,
            "fy": 710.132366,
            "cx": 640.0,
            "cy": 480.0,
            "s":  0
        }

    measurer = ClassicMeasurer(
        camera_height=1,
        height_thresh=0.01,
        intrinsics=intrinsics,
    )

    model = load_model(
        ckpt_dir=args.ckpt_dir,
        ckpt_type=args.ckpt_type,
        mode='segmentation',
    )

    cls = RFDETRClassifier(
        model=model,
        detection_threshold=0.5,
        roundness_threshold=0.8,
        lwh_min=(0.01, 0.01, 0.01),
        lwh_max=(0.45, 0.32, 0.32),
        measurer=measurer,
        margin_px=50,
        expand_px=0,
        fps=30,
        store_detections=True,
    )

    hdf_paths_sorted = sorted(list(args.ds_dir.glob('*.hdf5')), key=lambda x: int(x.stem))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    frames = []
    for idx, hdf_path in enumerate(tqdm(hdf_paths_sorted)):
        with h5py.File(hdf_path) as data:
            colors = data['colors'][()]
            depth = data['depth'][()]
        rgbd_img = np.dstack((colors, depth))
        _res = cls.step(rgbd_img)

        detections = cls.last_detections
        labels = []
        for det in detections:
            det_idx = det[4]
            if det_idx in _res:
                r = _res[det_idx]
                labels.append(
                    f'#{det_idx}\n'
                    f'l={r.length:.2f} w={r.width:.2f} h={r.height:.2f}\n'
                    f'roundness={r.roundness:.2f}'
                )
            else:
                labels.append(
                    f'#{det_idx}\n'
                    f'l=-1 w=-1 h=-1\n'
                    f'roundness=-1'
                )

        annotated_frame = box_annotator.annotate(colors.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        annotated_frame = mask_annotator.annotate(annotated_frame, detections=detections)
        frames.append(annotated_frame)

    # Save video
    for fps in (5, 15, 30):
        video_path = video_dir / f'tracking_seg_{fps}fps.avi'
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
