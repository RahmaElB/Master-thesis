# Master-thesis: Deep Learning for Echocardiogram Video Analysis

MUSI (UIB) master's thesis, detecting cardiac dysfunction from
echocardiogram videos using deep learning, supervised by Jose María Buades
Rubio and Iván Cortés Fernández. Continuation of Iván's earlier work at
Universidad de la Rioja.

## What this is

Classifying echocardiogram videos as **normal** vs **abnormal** cardiac
function, based on ejection fraction (EF):

- EF ≥ 50 → Normal
- EF < 50 → Abnormal

Using the EchoNet-Dynamic dataset (10,030 echo videos with EF labels and
official train/val/test splits; 7,465 / 1,288 / 1,277).

Four models compared: a ResNet18 frame averaging baseline, ResNet18+LSTM,
R3D-18 (3D CNN), and Swin3D-T (video transformer, Kinetics-400 pretrained).

Beyond the initial 4 way comparison, the project also covers: dataset
distribution analysis, full evaluation outputs (confusion matrix/ROC/PR),
robustness to random seed, data augmentation, robustness to image quality
degradation, sensitivity to the EF classification threshold, a per sample
error/"grey zone" analysis, and an exploratory EF regression model. See
[Experiments](#experiments-phases-1-9) below.

## Development environment

- **Cluster**: CSC Roihu (migrated from Puhti mid-project - see note below)
- **Python**: 3.12 (via CSC's `python-data` module for CPU-only scripts;
  Apptainer container for anything using PyTorch)
- **PyTorch**: 2.10.0+cu130 (delivered as an Apptainer container,
  `containers/pytorch_2.10_cuda13_roihu.sif` - Roihu has no plain `pytorch`
  module, PyTorch is only available this way)
- **CUDA**: 13.0
- **GPU**: NVIDIA GH200 (Grace Hopper, 120GB), via Slurm partition
  `gpumedium`, gres `gpu:gh200:1`
- **Key libraries**: torchvision, opencv-python-headless, pandas, numpy,
  scikit-learn, scipy, matplotlib (see `requirements.txt`)

**Note on Puhti → Roihu migration**: this project started on CSC Puhti
(NVIDIA V100 GPUs, plain `module load pytorch`) and was migrated to CSC
Roihu partway through (GH200 GPUs, ARM/x86 split login nodes, PyTorch only
via Apptainer container). All `scripts/*.sh` reflect the current Roihu
setup. `results/history/` contains some early runs from the Puhti era
(pre-refactor, uniform frame sampling).

## Repo layout

```
src/
  data/          # video I/O, frame sampling (uniform vs clip), augmentation,
                 # quality degradation, EDA script
  models/        # the 4 model definitions
  engine/        # train/eval loop, LR scheduling, metrics, plotting, wandb wrapper
  interpret/     # Grad-CAM + occlusion sensitivity
  train.py               # entry point: python -m src.train --model {baseline,cnn_lstm,r3d,swin3d}
  train_regression.py    # EF regression (continuous target, R3D-18 backbone)
  evaluate.py             # full-metric evaluation of a trained checkpoint (Acc/Prec/Rec/Spec/F1/AUC/CM/ROC/PR)
  evaluate_quality.py     # image-quality degradation robustness sweep
  evaluate_ef_threshold.py # EF threshold sensitivity sweep (40-70%)
  analyze_grey_zone.py    # per sample error analysis / EF "grey zone"
  analyze_regression.py   # EF regression results analysis
  compare_models.py       # combined ROC/PR/summary table across all 4 architectures
  aggregate_seeds.py      # multi-seed mean +/- std aggregation

scripts/                 # sbatch scripts for CSC Roihu
  run_baseline.sh, run_cnn_lstm.sh, run_r3d.sh, run_swin3d.sh
  run_eda.sh, run_interpretability.sh
  evaluation/             # per-checkpoint full-metric evaluation jobs
  robustness_seeds/       # multi-seed training jobs
  augmentation/           # augmented training jobs
  quality_robustness/     # quality degradation evaluation jobs
  regression/             # EF regression training job

notebooks/
  echonet_baseline_cpu.ipynb   # early exploratory notebook (CPU-friendly quick baseline)
  results_analysis.ipynb        # plots/comparisons, already run - open and read, no GPU or dataset needed

results/
  training/        # *_history.csv / *_test.csv per model, including seed/augmentation/regression variants
  eda/              # video length + EF distribution, class balance
  evaluation/       # full metrics, confusion matrices, ROC/PR curves, per-sample predictions
  robustness/       # multi-seed mean +/- std summaries
  ef_threshold/     # EF threshold sensitivity sweep results
  grey_zone/        # per-sample error analysis / grey-zone plots
  quality_robustness/  # image-quality degradation sweep results
  regression/       # EF regression analysis (true vs. predicted EF, error vs. EF)
  interpretability/ # Grad-CAM (R3D) + occlusion sensitivity (Swin3D) overlays
  history/          # early Puhti-era runs (pre-refactor, uniform sampling) - kept as historical record

logs/               # slurm .out logs

checkpoints/        # trained model weights - NOT in git (see .gitignore), regenerate via scripts/
containers/         # Apptainer .sif image - NOT in git (9GB), pull via the command in Setup below
```

`notebooks/results_analysis.ipynb` has already been run, all the plots and
tables are saved in it, so it can just be read top to bottom without needing
the dataset, a GPU, or the model checkpoints.

## Setup (CSC Roihu)

```bash
export ECHO_DATA_ROOT=/scratch/project_2018481/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/thesis_project6
export PYTHONPATH=$ECHO_PROJECT_ROOT
cd $ECHO_PROJECT_ROOT

# Pull the PyTorch container once (not tracked in git - ~9GB)
mkdir -p containers
apptainer pull containers/pytorch_2.10_cuda13_roihu.sif \
    docker://satama.csc.fi/r_installation_aida/pytorch:2.10_cuda13_roihu
```

Every `scripts/*.sh` job wraps its Python call in
`apptainer exec --nv containers/pytorch_2.10_cuda13_roihu.sif python3 ...`.
CPU-only scripts (EDA, comparison/aggregation scripts that just read saved
CSVs) instead use `module load python-data/3.12-31.03` and run directly on
the login node, no GPU or container needed.

## Current results (4-way model comparison)

| model     | test AUC | test acc | test F1 | epochs |
|-----------|----------|----------|---------|--------|
| baseline  | 0.908    | 0.825    | 0.879   | 15 |
| CNN+LSTM  | 0.900    | 0.861    | 0.915   | 15 |
| **R3D**   | **0.934**| 0.874    | 0.917   | 12 |
| Swin3D    | 0.912    | 0.869    | 0.916   | 15 |

R3D-18 is the best model overall, Swin3D-T a close second once its training
collapse was fixed (see Discussion in the thesis for the fix), both clearly
ahead of the frame-averaging baseline and CNN+LSTM - though CNN+LSTM's
headline accuracy is misleading (see Phase 3 below).

## Experiments (Phases 1-9)

The later phases reuse the same trained checkpoints where possible. 
Most require only additional evaluation or analysis of saved predictions rather than retraining.

1. **Baseline reproducibility** (`src/evaluate.py`): full documented
   metric set (Accuracy/Precision/Recall/Specificity/F1/ROC-AUC/confusion
   matrix) plus a "frozen record" JSON of every hyperparameter, for all 4
   models. -> `results/evaluation/*_frozen_record.json`

2. **Dataset analysis** (`src/data/eda.py`), EF distribution (skewness,
   normality test), class balance at multiple thresholds, video-length
   distribution, sampling coverage. -> `results/eda/`. EF is notably
   left-skewed (skewness -1.33); at the 50% threshold the dataset is 22.4%
   Abnormal / 77.6% Normal.

3. **Complete evaluation outputs** (`src/engine/plots.py`,
   `src/compare_models.py`): ROC curves, PR curves, confusion-matrix
   heatmaps per model, plus a combined 4-model comparison.
   -> `results/evaluation/`. Finding: CNN+LSTM has the highest raw accuracy
   but by far the weakest sensitivity to Abnormal cases (0.49), a case
   study in why accuracy alone is misleading here.

4. **Seed robustness** (`src/aggregate_seeds.py`), each model retrained
   with 2 additional random seeds (3 total), reporting mean +/- std.
   -> `results/robustness/`. AUC is very stable across seeds (std
   0.002-0.006 for all 4 models); Accuracy/F1 are noticeably less stable
   for the baseline specifically (std ~0.02).

5. **Data augmentation** (`src/data/augmentation.py`), small rotation,
   random-resized-crop, brightness/contrast jitter, applied only to the
   training split. No flipping (chamber left/right layout is diagnostically
   meaningful in an A4C view). -> `results/training/*_aug_test.csv`.
   Helped 3/4 models modestly. For CNN+LSTM, Accuracy/F1 decreased substantially while AUC stayed almost unchanged, suggesting an effect on the classification threshold rather than the ranking of predictions.

6. **Image-quality robustness** (`src/data/degradation.py`,
   `src/evaluate_quality.py`), controlled, deterministic degradation
   (blur+resolution loss+noise together, 4 severity levels) applied at
   evaluation time to already-trained checkpoints. -> `results/quality_robustness/`.
   Finding: Baseline/CNN+LSTM/R3D-18 all collapse toward predicting the
   majority class ("Normal") under degradation (sensitivity_abnormal drops
   toward 0.15-0.31); Swin3D-T fails in the *opposite* direction, becoming
   less accurate than a trivial majority-class baseline under
   moderate/severe degradation.

7. **EF threshold sensitivity** (`src/evaluate_ef_threshold.py`):
   re-derives ground truth at thresholds 40-70% from the already-saved
   predicted probabilities (no retraining). -> `results/ef_threshold/`.
   ROC-AUC decreases *monotonically* from 40% to 70% for every model, no
   peak at 50%; models are best at detecting severely reduced EF and
   weakest distinguishing mildly-reduced from normal.

8. **Error analysis / EF "grey zone"** (`src/analyze_grey_zone.py`), bins
   per-sample errors by distance from the 50% boundary.
   -> `results/grey_zone/`. Error rate is ~40% for videos within 5 EF points of the boundary, compared with <10% (often <2%) for videos 15+ points away. This pattern appears for all four models.

9. **EF regression** (`src/train_regression.py`,
   `src/analyze_regression.py`); R3D-18 backbone, continuous EF target,
   MSE loss. -> `results/regression/`. Test MAE=4.83, RMSE=6.63, R²=0.706
   (in line with published EchoNet-Dynamic regression benchmarks). A
   classifier derived by thresholding the regression output at 50% matches
   or slightly exceeds the purpose-trained R3D-18 classifier on
   Accuracy/F1/AUC, though with lower sensitivity to Abnormal cases.
   Regression error does not show the same concentration around the 50% boundary (p=0.15). This suggests that part of the grey-zone effect observed in classification may come from converting continuous EF values into binary labels.