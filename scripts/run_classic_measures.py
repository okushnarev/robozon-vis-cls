import sys
from pathlib import Path

import pandas as pd

# Add project root to PATH
if (project_root := str(Path.cwd())) not in sys.path:
    sys.path.append(project_root)

import json
from argparse import ArgumentParser, RawTextHelpFormatter, Namespace
from typing import Any

import h5py
import numpy as np

from src.measure.classic_measure import depth_mask, get_lwh, get_roundness


def parse_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--input-dir', '-i', type=Path, default=Path('datasets'),
                        help='Path to the directory containing the .hdf5 files\n'
                             'If no --filename (-f) specified looks recursively for .hdf5 files\n'
                             'Last directory name is used as a group name\n'
                             'Example:\n'
                             '      for -id datasets/gazebo\n'
                             '      found datasets/gazebo/pen/0.hdf5\n'
                             '      group name is pen')
    parser.add_argument('--output-dir', '-o', type=Path, default=Path('output'),
                        help='Path to the directory to save the output files')
    parser.add_argument('--filename', '-f', type=str, default=None,
                        help='Name of the .hdf5 file\n'
                             'If None|Omitted looks for all filenames')
    parser.add_argument('--out-csv', type=str, default=None,
                        help='Write output to CSV file to --output-dir (-o)')
    parser.add_argument('--print-output', action='store_true',
                        help='Print the output to stdout')

    cam_args = parser.add_argument_group('Camera arguments')
    cam_args.add_argument('--intrinsics', type=Path, default=None, required=True,
                          help='Path to the camera intrinsics file\n'
                               'Expected a .json file with fx, fy, cx, cy, s keys\n')
    cam_args.add_argument('--camera-height', type=float, default=1000,
                          help='Height of the camera (in mm) above target plane')
    cam_args.add_argument('--height-threshold', type=float, default=10,
                          help='Threshold (in mm) for depth object masking')

    return parser.parse_args()


def get_measurements(p: Path, intrinsics: dict[str, float], cfg: Namespace) -> dict[str, Any]:
    with h5py.File(p, 'r') as data:
        depth: np.ndarray = data['depth'][:] * 1000  # mm

    im_h, im_w = depth.shape

    valid_mask = depth_mask(depth, cfg.camera_height, cfg.height_threshold)

    l, w, h = get_lwh(
        mask=valid_mask,
        depth=depth,
        intrinsics=intrinsics,
        camera_height=cfg.camera_height,
        image_h=im_h,
        image_w=im_w,
    )

    # roundness
    r_ins_px, r_enc_px, roundness = get_roundness(valid_mask, verbose=True)

    return {
        'group':     p.parent.name,
        'filename':  p.stem,
        'length':    l,
        'width':     w,
        'height':    h,
        'roundness': roundness,
        'r_enc_px':  r_enc_px,
        'r_ins_px':  r_ins_px,
    }


def main():
    args = parse_args()

    with open(args.intrinsics, 'r') as intrinsics_file:
        intrinsics = json.load(intrinsics_file)

    # Resolve paths
    if args.filename:
        if (p := args.input_dir / f'{args.filename}.hdf5').exists():
            paths = [p]
        else:
            raise FileNotFoundError(f'No such file: {p}')
    else:
        paths = list(args.input_dir.rglob('*.hdf5'))
        if not paths:
            raise FileNotFoundError(f'Cannot find any .hdf5 files in {args.input_dir}')

    # Find measurements
    outputs: list[dict[str, Any]] = [get_measurements(path, intrinsics, args) for path in paths]

    # Resolve output
    if args.out_csv:
        df = pd.DataFrame(outputs)
        args.output_dir.mkdir(exist_ok=True, parents=True)
        csv_path = args.output_dir / f'{args.out_csv}.csv'
        df.to_csv(csv_path, index=False)
        print(f'Successfully saved to {csv_path}')

    if args.print_output:
        for output in outputs:
            print(output)


if __name__ == '__main__':
    main()
