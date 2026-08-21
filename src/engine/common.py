import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_weights(train_labels: np.ndarray) -> torch.Tensor:
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
    return torch.tensor(weights, dtype=torch.float32)


def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Warm up the learning rate linearly, then decay it with a cosine schedule.."""
    warmup_steps = max(1, min(warmup_steps, total_steps - 1)) if total_steps > 1 else 1

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, progress)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    grad_clip: float = None,
    grad_accum_steps: int = 1,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        x = batch["video"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * grad_accum_steps * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())

    # Apply the final optimizer step if the epoch ends before completing an accumulation cycle
    if len(loader) % grad_accum_steps != 0:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, epoch_f1


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Evaluate the model on one full validation pass and return loss, accuracy, F1, and AUC."""
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["video"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        epoch_auc = float("nan")

    return epoch_loss, epoch_acc, epoch_f1, epoch_auc


def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run a single inference pass over an entire loader and return per-sample
    arrays instead of aggregate metrics. This is the shared building block
    for:
      - Phase 1: full-metric "frozen" evaluation of a trained checkpoint
        (src/evaluate.py + src/engine/metrics.py)
      - Phase 7: EF threshold sensitivity sweep, re-threshold these same
        saved probabilities at 40/45/.../70% without re-running the model
      - Phase 8: error-vs-EF / grey zone analysis, needs exactly this
        (filename, true EF, true label, predicted probability) table

    Deliberately returns raw probabilities rather than thresholded
    predictions, so any threshold can be applied afterwards.
    """
    model.eval()
    filenames: List[str] = []
    efs: List[float] = []
    labels: List[int] = []
    probs: List[float] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["video"].to(device, non_blocking=True)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]

            filenames.extend(batch["filename"])
            efs.extend(batch["ef"].cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
            probs.extend(p.cpu().numpy().tolist())

    return {
        "filename": np.array(filenames),
        "ef": np.array(efs, dtype=float),
        "label": np.array(labels, dtype=int),
        "prob": np.array(probs, dtype=float),
    }