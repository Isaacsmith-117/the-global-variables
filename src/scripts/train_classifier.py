"""Train a small reason layer, calibrate on validation bottles, evaluate explicitly."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .classifier import (IMAGE_SIZE, REASONS, BottleClassifier, ReasonModel, bottle_scores,
                             load_dataset, masked_loss, read_crop, targets_and_masks)
    from .config import DEVICE_CHOICES, RUNS_DIR, select_device
else:
    from classifier import (IMAGE_SIZE, REASONS, BottleClassifier, ReasonModel, bottle_scores,
                            load_dataset, masked_loss, read_crop, targets_and_masks)
    from config import DEVICE_CHOICES, RUNS_DIR, select_device


@torch.no_grad()
def extract_features(model, root, rows, device):
    """Calculate frozen features once per crop, not again every training epoch."""
    features = []
    model.to(device).eval()
    for start in range(0, len(rows), 32):
        images = torch.stack([read_crop(root / r["image_path"]) for r in rows[start:start + 32]])
        features.append(model.backbone(images.to(device)).cpu())
    return torch.cat(features)


def confusion(truth, predicted):
    pairs = [(t, p) for t, p in zip(truth, predicted) if t is not None]
    tp = sum(t == 1 and p for t, p in pairs)
    fn = sum(t == 1 and not p for t, p in pairs)
    fp = sum(t == 0 and p for t, p in pairs)
    tn = sum(t == 0 and not p for t, p in pairs)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "positives": tp + fn, "negatives": fp + tn, "unknown": len(truth) - len(pairs),
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None}


def calibrate(bottles, enabled):
    """Choose per-reason validation thresholds by F2, favouring recall over precision."""
    thresholds = dict.fromkeys(REASONS)
    for reason in enabled:
        j = REASONS.index(reason)
        truth = [b["truth"][j] for b in bottles]
        if 1 not in truth or 0 not in truth:
            raise ValueError(f"Validation needs positive and negative bottles for {reason}.")
        best = (-1, -1)
        for step in range(1, 20):
            threshold = step / 20
            c = confusion(truth, [b["scores"][j] >= threshold for b in bottles])
            denominator = 5 * c["tp"] + 4 * c["fn"] + c["fp"]
            f2 = 5 * c["tp"] / denominator if denominator else 0
            # For equal F2 choose the higher, less permissive threshold.
            if (f2, threshold) > best:
                best = (f2, threshold)
        thresholds[reason] = best[1]
    return thresholds


def evaluate_bottles(bottles, thresholds, enabled):
    reasons = {}
    for reason in enabled:
        j = REASONS.index(reason)
        reasons[reason] = confusion([b["truth"][j] for b in bottles],
                                    [b["scores"][j] >= thresholds[reason] for b in bottles])
    rejected = [any(b["scores"][REASONS.index(r)] >= thresholds[r] for r in enabled)
                for b in bottles]
    return {"bottles": len(bottles), "per_reason": reasons,
            "plant_agreement": confusion([int(b["plant_status"] == "NOK") for b in bottles], rejected),
            "unchecked_reasons": [r for r in REASONS if r not in enabled],
            "scope": "Visual checks only; plant NOK can include unsupported/nonvisual reasons."}


def train(args):
    if args.epochs < 1:
        raise ValueError("Epochs must be positive.")
    metadata, rows = load_dataset(args.data)
    enabled = metadata["enabled_reason_outputs"]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    if not train_rows or not val_rows:
        raise ValueError("Training and validation splits must both contain bottles.")
    targets, masks = targets_and_masks(train_rows, enabled)
    val_targets, val_masks = targets_and_masks(val_rows, enabled)
    positives = (targets * masks).sum(0)
    negatives = ((1 - targets) * masks).sum(0)
    for reason in enabled:
        j = REASONS.index(reason)
        if positives[j] == 0 or negatives[j] == 0:
            raise ValueError(f"Training needs positive and negative crops for {reason}.")
    # Validate calibration coverage before downloading weights or creating output.
    val_truth = bottle_scores(val_rows, torch.zeros(len(val_rows), 3))
    calibrate(val_truth, enabled)
    print("Exploratory baseline: inspect distinct positive-bottle counts before claiming accuracy.")
    print("Enabled reasons:", ", ".join(enabled))
    if args.output.exists():
        raise FileExistsError(f"Output already exists; choose a new --output: {args.output}")
    torch.manual_seed(0)
    model = ReasonModel(pretrained=True)
    print("Extracting frozen training/validation features once...", flush=True)
    features = extract_features(model, args.data, train_rows, args.device)
    val_features = extract_features(model, args.data, val_rows, args.device)
    model.cpu()  # The tiny linear layer is inexpensive to train on cached CPU features.
    weights = negatives / positives.clamp_min(1)
    loader = DataLoader(TensorDataset(features, targets, masks), batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.001)
    best_loss, best_head, best_epoch = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, mask in loader:
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            loss = masked_loss(model.head(x), y, mask, weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = masked_loss(model.head(val_features), val_targets, val_masks, weights).item()
        if val_loss < best_loss:
            best_loss, best_head, best_epoch = val_loss, copy.deepcopy(model.head.state_dict()), epoch
        print(f"Epoch {epoch}/{args.epochs} | validation loss {val_loss:.4f}", flush=True)
    if best_head is None:
        raise ValueError("No finite validation checkpoint.")
    model.head.load_state_dict(best_head)
    with torch.no_grad():
        bottles = bottle_scores(val_rows, model.head(val_features).sigmoid())
    thresholds = calibrate(bottles, enabled)
    state = {"model_state": model.state_dict(), "reason_order": list(REASONS),
             "image_size": IMAGE_SIZE, "enabled_reasons": enabled, "thresholds": thresholds,
             "dataset_version": metadata["dataset_version"], "best_epoch": best_epoch,
             "seed": 0, "epochs": args.epochs}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(state, args.output / "classifier.pt")
    report = evaluate_bottles(bottles, thresholds, enabled)
    report.update({"split": "validation", "thresholds": thresholds, "best_epoch": best_epoch,
                   "note": "Used for checkpoint/threshold selection; not independent test results."})
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output / 'classifier.pt'}; test images were not used.")
    return report


def evaluate(args):
    _, rows = load_dataset(args.data)
    rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError(f"Empty evaluation split: {args.split}")
    classifier = BottleClassifier(args.checkpoint, args.device)
    features = extract_features(classifier.model, args.data, rows, args.device)
    classifier.model.cpu()
    with torch.no_grad():
        bottles = bottle_scores(rows, classifier.model.head(features).sigmoid())
    report = evaluate_bottles(bottles, classifier.thresholds, classifier.enabled)
    report["split"] = args.split
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--output", type=Path, default=RUNS_DIR / "classifier" / "baseline")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    for command in (training, evaluation):
        command.add_argument("--data", type=Path, required=True, help="Prepared dataset folder.")
        command.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    args = parser.parse_args(argv)
    args.device = select_device() if args.device == "auto" else args.device
    return train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
