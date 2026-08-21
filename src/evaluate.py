"""
Phase 1: freeze and document the baseline experiment, 
and provide the same evaluation setup for the other architectures. 

Loads a trained checkpoint and runs one inference pass over the selected split. 
The evaluation saves: 

1. <run_name>_<split>_frozen_record.json Hyperparameters 
and evaluation metrics for the experiment, including architecture, 
pretrained weights, input resolution, frames, temporal sampling, batch size, 
epochs, learning rate, optimizer, weight decay, dropout, loss, scheduler, EF threshold, 
classification threshold, random seed, accuracy, precision, recall, specificity, F1, 
ROC-AUC, and confusion matrix counts. 

2. <run_name>_<split>_confusion_matrix.csv 
The 2x2 confusion matrix with Abnormal/Normal labels. 

3. <run_name>_<split>_predictions.csv 
Per-sample filename, EF, true label, predicted probability, predicted label, and correctness. 
These predictions are also reused later for the EF threshold and grey-zone analyses without 
running inference again. 

4. <run_name>_<split>_roc_curve.png, _pr_curve.png, _confusion_matrix.png ROC, 
Precision-Recall, and confusion matrix plots used in Phase 3. The same plotting functions are reused by 
src/compare_models.py for the combined architecture comparison.


Usage (baseline, matching scripts/run_baseline.sh's hyperparameters):
    python -m src.evaluate --model baseline \
        --checkpoint checkpoints/baseline_64f_112px_clipp1_pretrained_best.pt \
        --num_frames 64 --img_size 112 --temporal_sampling clip --clip_period 1 \
        --batch_size 8 --epochs 15 --lr 1e-4 --weight_decay 1e-4 --dropout 0.3 \
        --split test --out_dir results/evaluation

The same evaluation can be run for cnn_lstm, r3d, and swin3d using the corresponding checkpoint 
and training configuration.
"""

import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EchoVideoDataset, make_splits
from src.models import MODEL_REGISTRY, MODEL_FAMILY, MODEL_NORMALIZE
from src.engine.common import predict_all
from src.engine.metrics import compute_metrics
from src.engine.plots import plot_roc_curve, plot_pr_curve, plot_confusion_matrix


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    p.add_argument("--checkpoint", type=str, required=True)

    p.add_argument("--data_root", type=str,
                    default=os.environ.get("ECHO_DATA_ROOT", "/scratch/project_2018481/echonet"))
    p.add_argument("--project_root", type=str,
                    default=os.environ.get("ECHO_PROJECT_ROOT", "/scratch/project_2018481/thesis_project6"))

    # Must match how the checkpoint was actually trained, copy these from
    # the matching scripts/run_*.sh, they are NOT re-derived automatically.
    p.add_argument("--num_frames", type=int, required=True)
    p.add_argument("--img_size", type=int, required=True)
    p.add_argument("--temporal_sampling", choices=["uniform", "clip"], default="clip")
    p.add_argument("--clip_period", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--ef_threshold", type=float, default=50.0,
                    help="EF cut used to build ground-truth labels (the one the model was trained on)")
    p.add_argument("--classification_threshold", type=float, default=0.5,
                    help="probability cutoff used to turn model output into a class prediction")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--seed", type=int, default=42)

    # These fields are only used in the frozen experiment record. 
    # They are not stored in the checkpoint, so they need to match 
    # the configuration used for the training run. Defaults are from the baseline setup.
    p.add_argument("--optimizer_desc", type=str, default="Adam")
    p.add_argument("--loss_desc", type=str, default="CrossEntropyLoss (class-weighted, inverse frequency)")
    p.add_argument("--scheduler_desc", type=str, default="None")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")

    p.add_argument("--out_dir", type=str, default="results/evaluation")
    p.add_argument("--run_name", type=str, default=None,
                    help="defaults to the checkpoint filename without extension")
    return p.parse_args()


def build_model_for_eval(args):
    model_cls = MODEL_REGISTRY[args.model]
    kwargs = dict(pretrained=False, dropout=args.dropout, num_classes=2)
    if args.model == "cnn_lstm":
        # cnn_lstm-specific defaults used in train.py
        kwargs.update(hidden_size=256, num_layers=1, bidirectional=False)
    return model_cls(**kwargs)


def main():
    args = get_args()
    run_name = args.run_name or os.path.splitext(os.path.basename(args.checkpoint))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = os.path.join(args.data_root, "FileList.csv")
    video_dir = os.path.join(args.data_root, "Videos")
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = make_splits(df)
    split_df = {"train": train_df, "val": val_df, "test": test_df}[args.split]

    model_family = MODEL_FAMILY[args.model]
    normalize = MODEL_NORMALIZE[args.model]

    dataset = EchoVideoDataset(
        df=split_df,
        video_dir=video_dir,
        num_frames=args.num_frames,
        img_size=args.img_size,
        ef_threshold=args.ef_threshold,
        model_family=model_family,
        normalize=normalize,
        temporal_sampling=args.temporal_sampling,
        clip_period=args.clip_period,
        split=args.split,  # "val"/"test" -> centered, reproducible sampling
        seed=args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model_for_eval(args).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Running inference on {len(dataset)} '{args.split}' videos for run '{run_name}'...")
    preds = predict_all(model, loader, device)

    metrics, cm = compute_metrics(preds["label"], preds["prob"], threshold=args.classification_threshold)
    print(json.dumps(metrics, indent=2))

    # per-sample predictions 
    pred_df = pd.DataFrame({
        "filename": preds["filename"],
        "ef": preds["ef"],
        "true_label": preds["label"],
        "pred_prob": preds["prob"],
        "pred_label": (preds["prob"] >= args.classification_threshold).astype(int),
    })
    pred_df["correct"] = pred_df["pred_label"] == pred_df["true_label"]
    pred_path = os.path.join(args.out_dir, f"{run_name}_{args.split}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved per-sample predictions to: {pred_path}")

    # confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["true_abnormal_0", "true_normal_1"],
        columns=["pred_abnormal_0", "pred_normal_1"],
    )
    cm_path = os.path.join(args.out_dir, f"{run_name}_{args.split}_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"Saved confusion matrix to: {cm_path}")

    # Phase 3: ROC curve, PR curve, confusion matrix heatmap 
    roc_path = os.path.join(args.out_dir, f"{run_name}_{args.split}_roc_curve.png")
    plot_roc_curve(preds["label"], preds["prob"], roc_path,
                    title=f"ROC Curve - {run_name}")
    print(f"Saved ROC curve to: {roc_path}")

    pr_path = os.path.join(args.out_dir, f"{run_name}_{args.split}_pr_curve.png")
    plot_pr_curve(preds["label"], preds["prob"], pr_path,
                  title=f"Precision-Recall Curve - {run_name}")
    print(f"Saved PR curve to: {pr_path}")

    cm_plot_path = os.path.join(args.out_dir, f"{run_name}_{args.split}_confusion_matrix.png")
    plot_confusion_matrix(cm, cm_plot_path, title=f"Confusion Matrix - {run_name}")
    print(f"Saved confusion matrix heatmap to: {cm_plot_path}")

    # frozen experiment record 
    record = {
        "run_name": run_name,
        "model_architecture": args.model,
        "checkpoint": args.checkpoint,
        "pretrained_weights": bool(args.pretrained),
        "input_resolution": args.img_size,
        "num_frames": args.num_frames,
        "temporal_sampling": args.temporal_sampling,
        "clip_period": args.clip_period if args.temporal_sampling == "clip" else None,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer": args.optimizer_desc,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "loss_function": args.loss_desc,
        "lr_scheduler": args.scheduler_desc,
        "ef_label_threshold": args.ef_threshold,
        "classification_threshold": args.classification_threshold,
        "random_seed": args.seed,
        "split_evaluated": args.split,
        "metrics": metrics,
    }
    record_path = os.path.join(args.out_dir, f"{run_name}_{args.split}_frozen_record.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved frozen experiment record to: {record_path}")


if __name__ == "__main__":
    main()