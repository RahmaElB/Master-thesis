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

Four models compared: a ResNet18 frame-averaging baseline, ResNet18+LSTM,
R3D-18 (3D CNN), and Swin3D-T (video transformer, Kinetics-400 pretrained).

## Repo layout

```
src/
  data/          # video I/O, frame sampling (uniform vs clip), EDA script
  models/        # the 4 model definitions
  engine/        # train/eval loop, LR scheduling, wandb wrapper
  interpret/     # Grad-CAM + occlusion sensitivity
  train.py       # single entry point: python -m src.train --model {baseline,cnn_lstm,r3d,swin3d}
scripts/         # sbatch scripts for Puhti (one per model + EDA + interpretability)
notebooks/
  echonet_baseline_cpu.ipynb   # early exploratory notebook (CPU-friendly quick baseline)
  results_analysis.ipynb        # all the plots/comparisons below, already run - open and read, no GPU or dataset needed
results/
  *_history.csv / *_test.csv   # current best run per model
  history/                      # runs from before I reorganized the code (uniform sampling)
  eda/                           # video length histogram / distribution, from the real dataset
  interpretability/               # Grad-CAM (R3D) + occlusion sensitivity (Swin3D) overlays
logs/            # slurm .out logs, matching results/
```

`notebooks/results_analysis.ipynb` has already been run, all the plots and
tables are saved in it, so it can just be read top to bottom without needing
the dataset, a GPU, or the model checkpoints.

## Current results

| model     | test AUC | test acc | test F1 | epochs |
|-----------|----------|----------|---------|--------|
| baseline  | 0.908    | 0.825    | 0.879   | 15 |
| CNN+LSTM  | 0.900    | 0.861    | 0.915   | 15 |
| **R3D**   | **0.934**| 0.874    | 0.917   | 12 |
| Swin3D    | 0.912    | 0.869    | 0.916   | 15 |

R3D is the best model right now, Swin3D a close second once its collapse was
fixed, both clearly ahead of the frame-averaging baseline and CNN+LSTM,
though the gap across all four ended up smaller than I expected going in.
All four use clip-based sampling with a random start every training epoch
(64 frames for baseline/CNN+LSTM/R3D, 32 for Swin3D).

## The R3D story

First attempt at porting R3D to the new clip-sampling code gave it the same
backbone/head learning-rate split I'd used to fix Swin3D (backbone at 1e-5,
head at 1e-4). Two 15-epoch runs with that setup both landed around **test
AUC 0.90-0.91**, worse than the original pre-refactor R3D result
(0.936-0.938), and both showed val_loss climbing steadily from ~0.6 to
1.3-1.5 while train accuracy raced to 0.98. Classic overfitting.

Turned out the LR split was the wrong fix for this model. Freezing R3D's
backbone down to 1e-5 was specifically what Swin3D needed because Swin3D was
*collapsing*, but R3D was never collapsing, so all that split did here was
leave the head to fit rapidly on top of barely-moving features, without the
backbone adapting alongside it to keep things regularized.

Dropped `--backbone_lr` entirely (single LR of 1e-4 for the whole network,
matching the original pre-refactor setup) and re-ran at 8 epochs as a check,
then 12:

| attempt | backbone LR split? | epochs | test AUC | val_loss trend |
|---------|--------------------|--------|----------|-----------------|
| 1 & 2   | yes (1e-5)         | 15     | 0.901-0.902 | 0.6 → 1.3-1.5 (exploding) |
| 3       | no                 | 8      | 0.923    | 0.52 → 0.63 (stable) |
| 4 (final) | no               | 12     | **0.934**| 0.47 → 0.61 (stable) |

Best epoch in the final run was epoch 7 (val_auc 0.937); epochs 8-12 hovered
around 0.92-0.93 without clearly beating it, so training is basically
plateaued by then. `scripts/run_r3d.sh` reflects the fix (no `--backbone_lr`,
`--epochs 12`). See `notebooks/results_analysis.ipynb` section 2 for the
before/after loss and AUC curves.

This result also isn't strictly apples-to-apples with the old 0.936-0.938
number, since this run also has random-start temporal augmentation (a
different 64-frame window each epoch), which the old uniform-sampling runs
didn't have. Landing back in the same range despite that extra variability
seems like a fine outcome, maybe a slightly more robust one, even without
being higher than before.

## The Swin3D story

The very first Swin3D run (before any of this reorganization) never learned
anything, val_acc/val_f1 identical every single epoch, val_auc stuck at
~0.55-0.59 (chance level). Diagnosis: LR of 1e-4 applied to the whole
pretrained transformer at once, no warmup, batch size 2, the pretrained
features got wrecked before the new head had learned anything to guide them.

Fix: backbone frozen for the first 2 epochs, separate LR for backbone
(1e-5) vs head (1e-3), linear warmup + cosine decay, gradient clipping,
effective batch size 16 via gradient accumulation. With that, val_auc climbs
cleanly from 0.68 (epoch 1, backbone still frozen) to 0.909 (best, epoch 11),
test AUC 0.912.

So the R3D and Swin3D fixes ended up being opposite prescriptions for
opposite problems: Swin3D needed its backbone protected with a much smaller
LR because it was collapsing; R3D needed the *opposite* (one LR for the
whole network) because it was already learning fine and the split just made
it overfit faster. Worth stating explicitly in the discussion, the lesson
isn't "transformers need X", it's "diagnose per-model, a fix that worked for
one architecture isn't automatically the right one for another."

## EDA: real dataset numbers (`results/eda/`)

Ran against the actual `FileList.csv` (10,030 videos):

- mean video length: 176.5 frames (std 57.9), min 28, max 1002
- only **0.57%** of videos have fewer than 64 frames
- at `num_frames=64, period=1` (what R3D/baseline/CNN+LSTM use), **99.4%**
  of videos are long enough for a genuinely full 64-frame span

So "go with 64" is a safe choice for this dataset specifically. This is also
the answer to why 32 frames looked competitive in the old uniform-sampling
runs: with uniform sampling the model always saw "the whole video,
compressed to 32 samples" regardless of true length, and most videos are on
the shorter side to begin with, with clip sampling that stops being true.

## Interpretability (`results/interpretability/`)

Ran both Grad-CAM (R3D) and occlusion sensitivity (Swin3D, since it has no
conv layer to hook Grad-CAM into) against the same 8 test videos, chosen
within 5 EF points of the 50 threshold, the genuinely hard cases.

| video | EF | true label | R3D | Swin3D |
|---|---|---|---|---|
| 0X6BA7CB2E44532208 | 54.7 | Normal | correct | correct |
| 0X608ED3201BB2F9A  | 53.0 | Normal | correct | correct |
| 0X1D393AED88F9D056 | 45.3 | Abnormal | correct | wrong (pred Normal) |
| 0X4A0C7C0FE8253F6E | 52.8 | Normal | correct | correct |
| 0X73CBCADA2191104C | 49.5 | Abnormal | wrong (pred Normal) | correct |
| 0X86643606EDB99E8  | 53.3 | Normal | correct | correct |
| 0X45357AC4B9F02268 | 45.4 | Abnormal | correct | correct |
| 0X3B80677CE0873E50 | 49.8 | Abnormal | wrong (pred Normal) | wrong (pred Normal) |

Both models get **6/8** right on this hard subset, same hit rate, but they
don't miss the same cases (R3D misses one, Swin3D misses a different one),
and only one video (0X3B80677CE0873E50, EF 49.8, the single closest case to
the threshold in the whole set) fools both. That complementary error pattern
is a reasonable argument for trying an ensemble of the two as future work,
even without time to build one now.

The two cases both models miss are worth a specific mention in the
discussion: both are Abnormal (EF just under 50) predicted as Normal, i.e.
false negatives right at the boundary, arguably the hardest kind of error
this task can produce, a couple of EF points on the wrong side of a
clinically somewhat arbitrary cutoff.

Heatmaps for both models qualitatively concentrate around the central
chamber region of the echo fan rather than scattering across the frame,
which is at least a sanity check that neither model is keying off scan
artifacts or the probe marker, but I'm not qualified to say whether it's
specifically tracking the left ventricle without a clinician's eye on it.
Worth showing to Iván.

## Weights & Biases

Working now, got a wandb.ai account set up, and since Puhti's compute nodes
don't have internet, training logs locally first and syncs automatically
once wandb is logged in. Project dashboard:
https://wandb.ai/elbouazzaouirahma8-bo-akademi/echonet-thesis

Only the R3D debugging runs (attempts 1, 3, 4 above) are logged there, since
baseline/CNN+LSTM/Swin3D all finished training before I had wandb set up.
Didn't think it was worth re-running those three (3.5-5.5 hours of V100 time
combined) purely to get them into wandb, since their full per-epoch numbers
already live in `results/*_history.csv` and are plotted in
`notebooks/results_analysis.ipynb` anyway, same information, no GPU cost.
The R3D dashboard on its own is genuinely useful though: the broken run's
train/val loss divergence is much easier to see as a live curve than
scrolling a CSV, which is part of how the LR-split diagnosis above got made.

## How to run things

```bash
pip install -r requirements.txt

export ECHO_DATA_ROOT=/scratch/project_2018481/relbouaz/echonet
export ECHO_PROJECT_ROOT=/scratch/project_2018481/relbouaz/thesis_project5
export PYTHONPATH=$ECHO_PROJECT_ROOT

python -m src.data.eda --csv_path $ECHO_DATA_ROOT/FileList.csv --out_dir results/eda

python -m src.train --model r3d --num_frames 64 --temporal_sampling clip \
    --clip_period 1 --lr 1e-4 --epochs 12 --grad_clip 1.0 --use_wandb

python -m src.interpret.run_interpretability --model r3d \
    --checkpoint checkpoints/r3d_64f_112px_clipp1_pretrained_best.pt \
    --num_frames 64 --img_size 112 --temporal_sampling clip --clip_period 1 \
    --num_examples 8 --out_dir results/interpretability

python -m src.interpret.run_interpretability --model swin3d \
    --checkpoint checkpoints/swin3d_32f_224px_clipp2_pretrained_best.pt \
    --num_frames 32 --img_size 224 --temporal_sampling clip --clip_period 2 \
    --num_examples 8 --out_dir results/interpretability
```

Or just use the sbatch scripts in `scripts/`, all of them already reflect
the fixed configs described above. Model checkpoints aren't included in this
zip (too large to share this way), `notebooks/results_analysis.ipynb` and
everything in `results/` and `logs/` don't need them, only re-training does.

## References

- Ouyang, D., He, B., Ghorbani, A., Lungren, M. P., Ashley, E. A., Liang, D.
  H., & Zou, J. Y. (2019). EchoNet-Dynamic: a large new cardiac motion video
  data resource for medical machine learning. *NeurIPS ML4H Workshop*.
- Magyar, B. et al. (2022). RVENet: A large echocardiographic dataset for the
  deep learning-based assessment of right ventricular function. *ECCV*.
- Bizopoulos, P., & Koutsouris, D. (2018). Deep learning in cardiology.
  *IEEE Reviews in Biomedical Engineering*, 12, 168-193.
- Patrianakos, A. P., Zacharaki, A. A., Skalidis, E. I., Hamilos, M. I.,
  Parthenakis, F. I., & Vardas, P. E. (2017). The growing role of
  echocardiography in interventional cardiology. *Hellenic Journal of
  Cardiology*, 58(1), 17-31.
- Sutanto, H. (2024). Transforming clinical cardiology through neural
  networks and deep learning. *Current Problems in Cardiology*, 49(4).
- Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). VideoMAE: Masked
  autoencoders are data-efficient learners for self-supervised video
  pre-training. *NeurIPS*.
- Ravi, N. et al. (2024). SAM 2: Segment anything in images and videos.
  *arXiv:2408.00714*.
- Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep
  networks via gradient-based localization. *ICCV*.
