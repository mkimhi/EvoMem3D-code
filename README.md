# EvoMem3D

Training and evaluation code for **EvoMem3D: Adaptive-Depth Temporal
Representation Memory for Multi-Camera 3D Detection**.

[Project page](https://mkimhi.github.io/EvoMem3D/) ·
[Model weights](https://huggingface.co/Kimhi/evomed3d-repdetr3d)

This release provides the EVA-02 DETR3D-family baseline and the final EvoMem3D
configuration recorded at **61.50 NDS / 52.76 mAP** on nuScenes validation.
The temporal-memory runtime retains adapters for RepDETR3D and StreamPETR;
the commands below use the two released RepDETR3D checkpoints.

## Installation

This implementation builds on **[SEPatch3D](https://github.com/Mingqj/SEPatch3D)**.
Please follow its [installation and data-preparation reference](https://github.com/Mingqj/SEPatch3D#get-started)
for the CUDA/OpenMMLab environment. The setup below uses our working versions:
Python 3.8, PyTorch 1.13.1, CUDA 11.7, MMCV 1.7.0, and MMDetection3D 1.0.0rc6.
An existing compatible SEPatch3D environment can be reused.

```bash
git clone https://github.com/mkimhi/EvoMem3D-code.git
cd EvoMem3D-code
conda env create -f environment.yml
conda activate evomem3d
pip install -r requirements.txt
git clone --depth 1 --branch v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
pip install --no-deps -e ./mmdetection3d
pip install flash-attn==1.0.9 --no-build-isolation
```

FlashAttention builds require a matching CUDA 11.7 toolkit/compiler, in addition
to the PyTorch CUDA runtime. Follow the SEPatch3D/StreamPETR installation guidance
for that dependency. Run all commands below from this repository root.

## nuScenes preparation

Prepare nuScenes `v1.0-trainval` as described by SEPatch3D, then generate our
temporal metadata:

```bash
mkdir -p data
ln -s /absolute/path/to/nuscenes data/nuscenes
python tools/create_data.py --root-path data/nuscenes
```

The converter expects the standard nuScenes samples, sweeps, maps, and metadata.
The detector itself consumes camera images. Existing compatible temporal info
files can be reused. The required layout is:

```text
data/nuscenes/
├── samples/
├── sweeps/
├── maps/
├── v1.0-trainval/
├── nuscenes2d_temporal_infos_train.pkl
└── nuscenes2d_temporal_infos_val.pkl
```

## Download weights

Both checkpoints are hosted on [Hugging Face](https://huggingface.co/Kimhi/evomed3d-repdetr3d).
Weights and datasets are not bundled in this repository.

```bash
python tools/download_weights.py
```

This creates `ckpts/repdetr3d_eva02_baseline.pth` and
`ckpts/evomem3d_repdetr3d_nds61_50.pth`. Use `--model baseline` or `--model final`
to download one checkpoint. The [HF metrics file](https://huggingface.co/Kimhi/evomed3d-repdetr3d/blob/main/evomem3d/metrics.json)
records the final run's NDS `0.61505` and mAP `0.52759`.

## Train

Train EvoMem3D from the provided full-depth detector baseline:

```bash
bash tools/dist_train.sh configs/tm_evomem3d.py 2 --work-dir work_dirs/tm_evomem3d
```

The released recipe uses two GPUs, two samples per GPU, seed 0, and 42,192
iterations. It initializes the detector from the baseline checkpoint, freezes
the pretrained detector, and trains the temporal modules and learned policy.
The config preserves the final run's optimizer, routing, refresh safeguards,
and policy normalization. Changing GPU count or batch size changes the recipe.

Resume an interrupted run using its full checkpoint:

```bash
bash tools/dist_train.sh configs/tm_evomem3d.py 2 \
  --work-dir work_dirs/tm_evomem3d --resume-from work_dirs/tm_evomem3d/latest.pth
```

Policy training uses full-encoder teacher features for its similarity target
and preceding-frame annotation summaries for teacher forcing. Evaluation uses
preceding detector predictions instead. No offline analysis CSV is required.

## Evaluate

Evaluate sequentially with **one GPU and one frame per batch**, preserving
scene order and temporal state:

```bash
# Final EvoMem3D model
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/tm_evomem3d.py \
  ckpts/evomem3d_repdetr3d_nds61_50.pth --eval mAP

# Full-depth baseline
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/tm_baseline.py \
  ckpts/repdetr3d_eva02_baseline.pth --eval mAP

# Your trained checkpoint
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/tm_evomem3d.py \
  work_dirs/tm_evomem3d/latest.pth --eval mAP
```

The test entry point loads the full checkpoint, including trained policy
parameters. The final model's internal L8/L12/L16 route names use inclusive
zero-based stop indices (9/13/17 executed blocks); Full executes 24. The release
preserves that trained behavior.

## Acknowledgments and license

We thank Mingqian Ji, Shanshan Zhang, and Jian Yang for
[SEPatch3D: Revisiting Token Compression for Accelerating ViT-based Sparse
Multi-View 3D Object Detectors](https://github.com/Mingqj/SEPatch3D), which provides
the upstream implementation and installation reference. We also acknowledge
StreamPETR, PETR/DETR3D, MMDetection3D, MMCV/MMDetection, EVA, and FlashAttention.
Original source notices are retained; see [NOTICE](NOTICE) and [LICENSE](LICENSE).
