# results/history

These are the runs from before I reorganized the code (the four separate
train_*.py scripts, uniform frame sampling only, no temporal augmentation).
Keeping them here for comparison so I don't lose the numbers I already
discussed with my supervisor.

Naming: `<model>_history.csv` = per-epoch train/val metrics, `<model>_test.csv`
(or `_test_results.csv`) = final test-set metrics for the checkpoint with the
best validation AUC.

Quick summary of what's in here (all with uniform sampling, i.e. num_frames
evenly spread across the *whole* video regardless of how long it is):

| model                        | frames | test AUC | test acc | notes |
|-------------------------------|--------|----------|----------|-------|
| baseline (ResNet18 + avg)     | 32     | 0.904    | 0.785    | |
| CNN+LSTM                      | 32     | 0.916    | 0.874    | |
| R3D-18                        | 16     | 0.920    | 0.835    | |
| R3D-18                        | 32     | 0.938    | 0.837    | best R3D result |
| R3D-18                        | 64     | 0.936    | 0.810    | slightly below 32f test AUC, but best val AUC of the three (0.947) |
| Swin3D-T (pretrained)         | 32     | 0.568    | 0.777    | **collapsed** - val_acc/val_f1 identical every epoch, model just predicts the majority class. Fixed now, see the main README results section and `src/models/swin3d.py` / `scripts/run_swin3d.sh`. |

Also note: the R3D checkpoint filenames from this batch of runs
(`r3d18_best.pt` / `r3d18_last.pt`) don't include the frame count, so the
16f/32f/64f runs actually overwrote each other's saved *model weights* -
only the CSV logs survived independently. That bug is fixed in the new
src/train.py (run names now always include num_frames, img_size and
sampling mode).

These numbers are all with uniform sampling. See the main `README.md` for
the updated clip-sampling + 64-frame results and the full before/after
comparison, including the R3D and Swin3D debugging stories.
