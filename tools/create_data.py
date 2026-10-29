"""Prepare the temporal nuScenes metadata used for training and evaluation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-path', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'])
    parser.add_argument('--max-sweeps', type=int, default=10)
    args = parser.parse_args()
    from tools.data_converter.nuscenes_converter import create_nuscenes_infos
    create_nuscenes_infos(args.root_path, 'nuscenes2d',
                          version=args.version, max_sweeps=args.max_sweeps)


if __name__ == '__main__':
    main()
