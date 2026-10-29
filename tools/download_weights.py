"""Download the two released checkpoints from Hugging Face."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['baseline', 'final', 'all'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('ckpts'))
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    import shutil
    files = {
        'baseline': ('baseline/repdetr3d_eva02_baseline.pth', 'repdetr3d_eva02_baseline.pth'),
        'final': ('evomem3d/evomem3d_repdetr3d_nds61_50.pth', 'evomem3d_repdetr3d_nds61_50.pth'),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (remote, local) in files.items():
        if args.model not in ('all', name):
            continue
        downloaded = hf_hub_download('Kimhi/evomed3d-repdetr3d', remote)
        target = args.output_dir / local
        shutil.copyfile(downloaded, target)
        print(f'{name}: {target}')


if __name__ == '__main__':
    main()
