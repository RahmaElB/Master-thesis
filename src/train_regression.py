"""
Phase 9 : EF regression instead of binary classification. 
This is kept separate from src/train.py because the classification training code is 
built around CrossEntropyLoss, softmax, and argmax, while regression uses a continuous EF target. 
R3D-18 is reused with num_classes=1, so the architecture itself does not need to change. 
The same video setup is used as in the R3D classification experiments: 64 frames, 112px, 
and clip sampling with period 1. The dataset already provides the continuous ef value for each sample, 
so regression trains directly on that field instead of the binary label. 

Loss: MSE. Reported metrics: MAE, RMSE, and R^2. After the final test evaluation, 
the script also saves filename, true_ef, pred_ef, and abs_error for the analysis in src/analyze_regression.py.

Usage (matches scripts/run_r3d.sh's data/video hyperparameters):
    python -m src.train_regression \
        --num_frames 64 --img_size 112 --temporal_sampling clip --clip_period 1 \
        --batch_size 4 --epochs 12 --lr 1e-4 --weight_decay 1e-4 --dropout 0.3 \
        --grad_clip 1.0
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import EchoVideoDataset, make_splits
from src.models.r3d import R3DClassifier
from src.engine.common import set_seed


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str,
                    default=os.environ.get("ECHO_DATA_ROOT", "/scratch/project_2018481/echonet"))
    p.add_argument("--project_root", type=str,
                    default=os.environ.get("ECHO_PROJECT_ROOT", "/scratch/project_2018481/thesis_project6"))

    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--img_size", type=int, default=112)
    p.add_argument("--temporal_sampling", choices=["uniform", "clip"], default="clip")
    p.add_argument("--clip_period", type=int, default=1)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                    help="train from scratch instead of ImageNet/Kinetics-pretrained weights")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    running_loss = 0.0
    all_errors = []
    for batch in loader:
        x = batch["video"].to(device, non_blocking=True)
        y = batch["ef"].to(device, non_blocking=True)

        pred = model(x).squeeze(1)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        all_errors.extend((pred - y).detach().cpu().numpy().tolist())

    all_errors = np.array(all_errors)
    epoch_loss = running_loss / len(loader.dataset)
    mae = np.abs(all_errors).mean()
    rmse = np.sqrt((all_errors ** 2).mean())
    return epoch_loss, mae, rmse


@torch.no_grad()
def evaluate_regression(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_true, all_pred, all_filenames = [], [], []
    for batch in loader:
        x = batch["video"].to(device, non_blocking=True)
        y = batch["ef"].to(device, non_blocking=True)

        pred = model(x).squeeze(1)
        loss = criterion(pred, y)
        running_loss += loss.item() * x.size(0)

        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_filenames.extend(batch["filename"])

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    epoch_loss = running_loss / len(loader.dataset)
    errors = all_pred - all_true
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    ss_res = (errors ** 2).sum()
    ss_tot = ((all_true - all_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return epoch_loss, mae, rmse, r2, all_filenames, all_true, all_pred


def main():
    args = get_args()
    set_seed(args.seed)

    run_name = args.run_name or f"r3d_regression_{args.num_frames}f_{args.img_size}px_" \
                                  f"{args.temporal_sampling}{args.clip_period if args.temporal_sampling=='clip' else ''}_pretrained"

    csv_path = os.path.join(args.data_root, "FileList.csv")
    video_dir = os.path.join(args.data_root, "Videos")
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = make_splits(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    def make_ds(split_df, split_name):
        return EchoVideoDataset(
            df=split_df, video_dir=video_dir,
            num_frames=args.num_frames, img_size=args.img_size,
            model_family="3d", normalize="tanh",
            temporal_sampling=args.temporal_sampling, clip_period=args.clip_period,
            split=split_name, seed=args.seed,
        )

    train_ds = make_ds(train_df, "train")
    val_ds = make_ds(val_df, "val")
    test_ds = make_ds(test_df, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = R3DClassifier(pretrained=args.pretrained, dropout=args.dropout, num_classes=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.join(args.project_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.project_root, "results"), exist_ok=True)
    best_path = os.path.join(args.project_root, "checkpoints", f"{run_name}_best.pt")
    last_path = os.path.join(args.project_root, "checkpoints", f"{run_name}_last.pt")

    history = []
    best_val_mae = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae, train_rmse = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=args.grad_clip)
        val_loss, val_mae, val_rmse, val_r2, *_ = evaluate_regression(
            model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f}, train_mae={train_mae:.3f}, "
              f"train_rmse={train_rmse:.3f} | val_loss={val_loss:.4f}, val_mae={val_mae:.3f}, "
              f"val_rmse={val_rmse:.3f}, val_r2={val_r2:.3f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_mae": train_mae,
                         "train_rmse": train_rmse, "val_loss": val_loss, "val_mae": val_mae,
                         "val_rmse": val_rmse, "val_r2": val_r2})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to: {best_path}")

    torch.save(model.state_dict(), last_path)
    print(f"Saved last model to: {last_path}")

    history_path = os.path.join(args.project_root, "results", f"{run_name}_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved history to: {history_path}")

    # final test evaluation using the BEST checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    print("\nEvaluating best model on test set...")
    test_loss, test_mae, test_rmse, test_r2, filenames, true_ef, pred_ef = evaluate_regression(
        model, test_loader, criterion, device)
    print(f"TEST | loss={test_loss:.4f}, MAE={test_mae:.3f}, RMSE={test_rmse:.3f}, R2={test_r2:.3f}")

    test_row = {
        "run_name": run_name, "model": "r3d_regression",
        "test_loss": test_loss, "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2,
        "num_epochs": args.epochs, "batch_size": args.batch_size, "num_frames": args.num_frames,
        "img_size": args.img_size, "temporal_sampling": args.temporal_sampling,
        "clip_period": args.clip_period, "lr": args.lr, "weight_decay": args.weight_decay,
        "dropout": args.dropout, "grad_clip": args.grad_clip, "pretrained": args.pretrained,
        "seed": args.seed, "total_runtime_sec": time.time() - start_time,
    }
    test_results_path = os.path.join(args.project_root, "results", f"{run_name}_test.csv")
    pd.DataFrame([test_row]).to_csv(test_results_path, index=False)
    print(f"Saved test results to: {test_results_path}")

    pred_df = pd.DataFrame({
        "filename": filenames,
        "true_ef": true_ef,
        "pred_ef": pred_ef,
        "abs_error": np.abs(pred_ef - true_ef),
    })
    pred_path = os.path.join(args.project_root, "results", f"{run_name}_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved per-sample regression predictions to: {pred_path}")


if __name__ == "__main__":
    main()