import json
import sys
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

if __name__ == '__main__':
    exp_name = '1'
    ckpt_dir = Path('checkpoints/seg_1/')
    ds_dir = Path('datasets/prep/gazebo_vid/') / exp_name
    video_dir = Path('videos') / ds_dir.parent.name / exp_name
    video_dir.mkdir(exist_ok=True, parents=True)

    intrinsics_path = Path('params/intrinsics_1280.json')
    with open(intrinsics_path, 'r') as intrinsics_file:
        intrinsics = json.load(intrinsics_file)

    measurer = ClassicMeasurer(
        camera_height=1,
        height_thresh=0.01,
        intrinsics=intrinsics,
    )

    ckpt_type = 'last.ckpt'
    model = load_model(
        ckpt_dir=ckpt_dir,
        ckpt_type=ckpt_type,
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

    hdf_paths_sorted = sorted(list(ds_dir.glob('*.hdf5')), key=lambda x: int(x.stem))

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
