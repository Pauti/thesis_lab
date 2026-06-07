# thesis_lab

Research codebase for my master's thesis comparing convolutional and transformer-based architectures for 3D brain tumor segmentation on BraTS 2021.

## Research Overview

Three research questions drive the experimental design:

- **RQ1 - Architecture Comparison**: How do nnU-Net (CNN) and UNETR (Vision Transformer) compare on 3D glioma segmentation?
- **RQ2 - Augmentation Sensitivity**: Which augmentation strategies yield the most robust models, and does sensitivity differ between architectures?
- **RQ3 - Multi-Task Learning**: Does an auxiliary ET-volume regression task improve segmentation through encoder regularization?

## Models

**nnU-Net** ([Isensee et al., Nature Methods 2021](https://www.researchgate.net/publication/346711234_nnU-Net_a_self-configuring_method_for_deep_learning-based_biomedical_image_segmentation)) - self-configuring CNN; auto-configured via `nnUNetv2_plan_and_preprocess`. Extended via custom Trainer subclasses in `models/nnunet_trainers.py` for augmentation ablation.

**UNETR** ([Hatamizadeh et al., WACV 2022](https://arxiv.org/abs/2103.10504)) - ViT-B/16 encoder + CNN decoder via MONAI. Configured through `configs/exp03_unetr.yaml` and trained with `scripts/train.py`.

## Dataset

BraTS 2021 Training Set - 1251 multi-parametric MRI cases (T1, T1ce, T2, FLAIR), 1mm isotropic, labels: NCR/NET (1), ED (2), ET (4). Evaluation regions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET).

**Split strategy**:
- RQ1 (EXP-02, EXP-03): 5-fold CV over all 1251 cases
- RQ2/RQ3 ablation: 125 fixed held-out test cases + 3-fold CV over remaining 1126 cases

## Experiment Grid

| ID | Architecture | Augmentation | Multi-Task | Folds | Epochs | Status |
|----|-------------|--------------|------------|-------|--------|--------|
| EXP-02 | nnU-Net | Default | No | 5 | 1000 | Completed |
| EXP-03 | UNETR | Default | No | 5 | 200+ES | Completed |
| ABL-01 | nnU-Net | Disabled | No | 3 | 100 | Completed |
| ABL-02 | nnU-Net | Reduced | No | 3 | 100 | In progress |
| ABL-03 | nnU-Net | Default | No | 3 | 100 | Completed |
| ABL-04 | nnU-Net | Extended | No | 3 | 100 | Completed |
| ABL-05 | UNETR | Disabled | No | 3 | 100 | Completed |
| ABL-06 | UNETR | Reduced | No | 3 | 100 | In progress |
| ABL-07 | UNETR | Default | No | 3 | 100 | Completed |
| ABL-08 | UNETR | Extended | No | 3 | 100 | Completed |
| EXP-04 | nnU-Net | Default | Yes | 3 | 100 | Planned |
| EXP-05 | UNETR | Default | Yes | 3 | 100 | Planned |

Augmentation tiers: **Disabled** (no transforms) / **Reduced** (spatial only) / **Default** (framework paper config) / **Extended** (Default + elastic deformation + extra intensity).

## Repository Structure

```
configs/          YAML configs for UNETR experiments
experiments/      Training outputs, checkpoints, logs
figures/          TikZ architecture diagrams
models/
  unetr.py              UNETR model factory (MONAI)
  nnunet_trainers.py    nnU-Net Trainer subclasses for augmentation ablation
plan/             Research documentation, planning files, CLAUDE.md
scripts/
  train.py              Main UNETR training loop
  run_exp02.sh          nnU-Net training runner
  run_exp03.sh          UNETR training runner
  run_ablation.sh       Ablation experiment runner (--model, --tier, --fold)
  analyze_exp02/03.py   Per-experiment result analysis
  plot_*.py             Convergence and comparison plots
  infer_ablation_*.py   Test-set inference scripts
utils/
  augmentation.py       4-tier MONAI augmentation pipeline
  losses.py             DiceCELoss
  metrics.py            BraTS Dice (WT/TC/ET) + HD95
  dataloader.py         BraTS dataset and split loading
```

## Setup

```bash
# PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Remaining dependencies
pip install -r requirements.txt

# nnU-Net environment variables (add to shell profile)
export nnUNet_raw="/data/paul_daten/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/data/paul_daten/nnunet/nnUNet_preprocessed"
export nnUNet_results="/data/paul_daten/nnunet/nnUNet_results"
```

See `scripts/setup_nnunet_env.sh` for full environment setup.

## Running Experiments

**nnU-Net baseline**:
```bash
bash scripts/run_exp02.sh
```

**UNETR baseline**:
```bash
bash scripts/run_exp03.sh
```

**Augmentation ablation** (model: nnunet or unetr, tier: disabled/reduced/default/extended):
```bash
bash scripts/run_ablation.sh --model unetr --tier default --fold 0
bash scripts/run_ablation.sh --model nnunet --tier disabled  # all folds sequentially
```

## Evaluation

Metrics are computed post-hoc via `utils/metrics.py` using the BraTS evaluation convention (Dice WT/TC/ET, HD95).

## References

- Isensee et al. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. https://www.researchgate.net/publication/346711234_nnU-Net_a_self-configuring_method_for_deep_learning-based_biomedical_image_segmentation
- Hatamizadeh et al. (2022). UNETR: Transformers for 3D Medical Image Segmentation. *WACV*. https://arxiv.org/abs/2103.10504
- Mlynarski et al. (2019). Deep learning with mixed supervision for brain tumor segmentation. *Journal of Medical Imaging*. https://doi.org/10.1117/1.JMI.6.3.034002
- Bakas et al. / BraTS 2021 Challenge organizers. https://arxiv.org/abs/2107.02314
