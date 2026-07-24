from argparse import ArgumentParser, RawDescriptionHelpFormatter, RawTextHelpFormatter
from pathlib import Path


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


def main():
    args = parse_args()


if __name__ == '__main__':
    main()
