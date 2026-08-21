"""
Single training entry point for all four models.

Originally I had separate training scripts for each model, 
even though most of the dataset loading and training/evaluation logic was the same. 
This also caused a checkpoint naming issue for R3D, where num_frames was missing 
from the path and runs with different frame counts could overwrite each other. 
I moved the shared training logic here and into src/engine, while keeping the model-specific parts in src/models.
The run configuration is also included in the output names/results so I can keep track of
 which setup produced each result.

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
from src.models import MODEL_REGISTRY, MODEL_FAMILY, MODEL_NORMALIZE
from src.engine.common import (
    set_seed,
    get_class_weights,
    train_one_epoch,
    evaluate,
    build_warmup_cosine_scheduler,
)
from src.engine.wandb_logger import WandbLogger


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)

    # Keep the original cluster paths as defaults, but allow overriding them
    # through environment variables or CLI when running somewhere else.
    p.add_argument("--data_root", type=str,
                    default=os.environ.get("ECHO_DATA_ROOT", "/scratch/project_2018481/relbouaz/echonet"))
    p.add_argument("--project_root", type=str,
                    default=os.environ.get("ECHO_PROJECT_ROOT", "/scratch/project_2018481/relbouaz/thesis_project"))

    # Default to 64 frames based on the supervisor feedback. Earlier 32 frame
    # experiments were quite competitive with uniform sampling, probably because
    # many videos are relatively short. With fixed length clip sampling, the
    # difference between 32 and 64 frames becomes more meaningful.
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--img_size", type=int, default=None,
                    help="defaults to 112, or 224 for swin3d if not set")
    p.add_argument("--temporal_sampling", choices=["uniform", "clip"], default="clip")
    p.add_argument("--clip_period", type=int, default=1,
                    help="only used when --temporal_sampling clip; frames are spaced this many real frames apart")

    # Optimization
    p.add_argument("--batch_size", type=int, default=None, help="defaults per model if not set")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=None,
                    help="if set, uses a separate (usually smaller) LR for pretrained backbone params")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--warmup_frac", type=float, default=0.0,
                    help="fraction of total optimizer steps used for linear LR warmup")
    p.add_argument("--freeze_epochs", type=int, default=0,
                    help="freeze pretrained backbone for this many epochs before unfreezing (r3d/swin3d only)")
    p.add_argument("--augment", action="store_true",
                    help="Phase 5: apply train-time augmentation (small rotation, "
                         "random-resized-crop, brightness/contrast jitter) - see "
                         "src/data/augmentation.py. Never applied to val/test.")

    # Model-specific
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.add_argument("--lstm_hidden_size", type=int, default=256)
    p.add_argument("--lstm_num_layers", type=int, default=1)
    p.add_argument("--lstm_bidirectional", action="store_true")

    # Misc
    p.add_argument("--ef_threshold", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="echonet-thesis")

    return p.parse_args()


def build_model(args):
    model_cls = MODEL_REGISTRY[args.model]
    if args.model == "baseline":
        return model_cls(pretrained=args.pretrained, dropout=args.dropout, num_classes=2)
    if args.model == "cnn_lstm":
        return model_cls(
            pretrained=args.pretrained,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            dropout=args.dropout,
            bidirectional=args.lstm_bidirectional,
            num_classes=2,
        )
    if args.model in ("r3d", "swin3d"):
        return model_cls(pretrained=args.pretrained, dropout=args.dropout, num_classes=2)
    raise ValueError(args.model)


def default_batch_size(model_name: str) -> int:
    return {"baseline": 8, "cnn_lstm": 4, "r3d": 4, "swin3d": 2}[model_name]


def main():
    args = get_args()
    set_seed(args.seed)

    if args.img_size is None:
        args.img_size = 224 if args.model == "swin3d" else 112
    if args.batch_size is None:
        args.batch_size = default_batch_size(args.model)

    checkpoint_dir = os.path.join(args.project_root, "checkpoints")
    results_dir = os.path.join(args.project_root, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(args.data_root, "FileList.csv")
    video_dir = os.path.join(args.data_root, "Videos")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Missing video folder: {video_dir}")

    df = pd.read_csv(csv_path)
    print("Metadata shape:", df.shape)
    train_df, val_df, test_df = make_splits(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    model_family = MODEL_FAMILY[args.model]
    normalize = MODEL_NORMALIZE[args.model]

    def make_ds(split_df, split_name):
        return EchoVideoDataset(
            df=split_df,
            video_dir=video_dir,
            num_frames=args.num_frames,
            img_size=args.img_size,
            ef_threshold=args.ef_threshold,
            model_family=model_family,
            normalize=normalize,
            temporal_sampling=args.temporal_sampling,
            clip_period=args.clip_period,
            split=split_name,
            seed=args.seed,
            # Apply augmentation only to training data. make_ds is also used for
            # validation/test, so check the split here even when --augment is enabled.
            augment=(args.augment and split_name == "train"),
        )

    train_dataset = make_ds(train_df, "train")
    val_dataset = make_ds(val_df, "val")
    test_dataset = make_ds(test_df, "test")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)

    train_labels = (train_df["EF"] >= args.ef_threshold).astype(int).values
    class_weights = get_class_weights(train_labels)
    print("Class weights:", class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    use_staged_finetune = args.model in ("r3d", "swin3d") and (args.freeze_epochs > 0 or args.backbone_lr is not None)
    if args.model in ("r3d", "swin3d") and args.freeze_epochs > 0:
        model.set_backbone_trainable(False) if hasattr(model, "set_backbone_trainable") else None
        print(f"Backbone frozen for the first {args.freeze_epochs} epoch(s).")

    param_groups = model.param_groups(lr=args.lr, backbone_lr=args.backbone_lr) \
        if hasattr(model, "param_groups") else [{"params": model.parameters(), "lr": args.lr}]
    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps) if warmup_steps > 0 else None

    weights_tag = "pretrained" if args.pretrained else "scratch"
    sampling_tag = f"{args.temporal_sampling}" + (f"p{args.clip_period}" if args.temporal_sampling == "clip" else "")
    aug_tag = "_aug" if args.augment else ""
    run_name = args.run_name or f"{args.model}_{args.num_frames}f_{args.img_size}px_{sampling_tag}_{weights_tag}{aug_tag}"

    best_model_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
    last_model_path = os.path.join(checkpoint_dir, f"{run_name}_last.pt")
    history_path = os.path.join(results_dir, f"{run_name}_history.csv")
    test_results_path = os.path.join(results_dir, f"{run_name}_test.csv")

    wandb_logger = WandbLogger(enabled=args.use_wandb, project=args.wandb_project,
                                run_name=run_name, config=vars(args))

    history = []
    best_val_auc = -1.0
    start_time = time.time()

    for epoch in range(args.epochs):
        if use_staged_finetune and epoch == args.freeze_epochs and hasattr(model, "set_backbone_trainable"):
            model.set_backbone_trainable(True)
            print(f"Epoch {epoch+1}: unfreezing backbone for fine-tuning.")

        epoch_start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, grad_clip=args.grad_clip, grad_accum_steps=args.grad_accum_steps,
        )
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "val_auc": val_auc,
            "lr_head": optimizer.param_groups[-1]["lr"],
            "epoch_time_sec": time.time() - epoch_start,
        }
        history.append(row)
        wandb_logger.log(row, step=epoch + 1)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, val_auc={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

    torch.save(model.state_dict(), last_model_path)
    print(f"Saved last model to: {last_model_path}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)
    print(f"Saved history to: {history_path}")

    print("Evaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion, device)

    test_row = {
        "run_name": run_name,
        "model": args.model,
        "test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1, "test_auc": test_auc,
        "best_val_auc": best_val_auc,
        "num_epochs": args.epochs, "batch_size": args.batch_size,
        "num_frames": args.num_frames, "img_size": args.img_size,
        "temporal_sampling": args.temporal_sampling, "clip_period": args.clip_period,
        "ef_threshold": args.ef_threshold, "lr": args.lr, "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay, "dropout": args.dropout,
        "grad_clip": args.grad_clip, "grad_accum_steps": args.grad_accum_steps,
        "warmup_frac": args.warmup_frac, "freeze_epochs": args.freeze_epochs,
        "pretrained": args.pretrained,
        "augment": args.augment,
        "total_runtime_sec": time.time() - start_time,
    }
    pd.DataFrame([test_row]).to_csv(test_results_path, index=False)

    print(f"TEST | loss={test_loss:.4f}, acc={test_acc:.4f}, f1={test_f1:.4f}, auc={test_auc:.4f}")
    print(f"Saved test results to: {test_results_path}")

    wandb_logger.log({"test_loss": test_loss, "test_acc": test_acc, "test_f1": test_f1, "test_auc": test_auc})
    wandb_logger.finish()


if __name__ == "__main__":
    main()