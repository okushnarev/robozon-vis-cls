import json
from pathlib import Path

import h5py
import numpy as np

from src.classifiers import RFDETRClassifier
from src.measure.measurers import ClassicMeasurer
from src.wrappers import load_model

if __name__ == '__main__':
    intrinsics_path = Path('params/intrinsics_1280.json')
    with open(intrinsics_path, 'r') as intrinsics_file:
        intrinsics = json.load(intrinsics_file)

    measurer = ClassicMeasurer(
        camera_height=1,
        height_thresh=0.07,
        intrinsics=intrinsics,
    )

    ckpt_dir = Path('checkpoints/default/')
    ckpt_type = 'checkpoint_best_total'
    model = load_model(
        ckpt_dir=ckpt_dir,
        ckpt_type=ckpt_type,
        mode='detection',
    )

    cls = RFDETRClassifier(
        model=model,
        detection_threshold=0.5,
        roundness_threshold=0.8,
        measurer=measurer,
        margin_px=50,
        expand_px=10,
        fps=30,
    )

    ds_dir = Path('datasets/prep/gazebo_vid/1/')
    hdf_paths_sorted = sorted(list(ds_dir.glob('*.hdf5')), key=lambda x: int(x.stem))

    for idx, hdf_path in enumerate(hdf_paths_sorted):
        with h5py.File(hdf_path) as data:
            colors = data['colors'][()]
            depth = data['depth'][()]
        rgbd_img = np.dstack((colors, depth))
        _res = cls.step(rgbd_img)
        print(idx, _res)
        print()
