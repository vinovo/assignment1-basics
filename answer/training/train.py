import argparse
import configparser
import json
import os
import time

import numpy as np
import torch

from answer.transformers.transformer_lm import TransformerLM
from answer.training.adamw import AdamW
from answer.training.cosine_lr import cosine_lr
from answer.training.data_loader import get_batch
from answer.training.cross_entropy import cross_entropy
from answer.training.gradient_clipping import clip_gradient
from answer.training.checkpoint import save_checkpoint

CHECKPOINT_INTERVAL = 100
EVAL_INTERVAL = 10
EVAL_ITERS = 10

# Schema: (section, cfg_key, type, default_or_None, [dict_key])
# When dict_key is provided, it overrides cfg_key as the output dict key.
CONFIG_SCHEMA = [
    # Experiment metadata
    ("experiment", "name", str, None),
    # Model hyperparameters
    ("model", "d_model", int, None),
    ("model", "num_heads", int, None),
    ("model", "d_ff", int, None),
    ("model", "vocab_size", int, None),
    ("model", "context_length", int, None),
    ("model", "num_layers", int, None),
    ("model", "theta", float, 10000.0),
    ("model", "eps", float, 1e-5),
    # Optimizer hyperparameters
    ("optimizer", "lr", float, 6e-4),
    ("optimizer", "beta1", float, 0.9),
    ("optimizer", "beta2", float, 0.999),
    ("optimizer", "eps", float, 1e-8, "optimizer_eps"),
    ("optimizer", "weight_decay", float, 0.01),
    # LR scheduler hyperparameters
    ("lr_scheduler", "lr_max", float, 6e-4),
    ("lr_scheduler", "lr_min", float, 6e-5),
    ("lr_scheduler", "warmup_iters", int, 1000),
    ("lr_scheduler", "cosine_iters", int, 10000),
    # Training hyperparameters
    ("training", "batch_size", int, 32),
    ("training", "max_iters", int, 10000),
    ("training", "grad_clip_max_norm", float, 1.0),
    ("training", "device", str, "cuda"),
]


def load_config(path: str) -> dict:
    """Load a .cfg file and return a flat dict of all hyperparameters."""
    parser = configparser.ConfigParser()
    parser.read(path)

    config = {}
    for entry in CONFIG_SCHEMA:
        section, cfg_key, typ, default = entry[:4]
        dict_key = entry[4] if len(entry) > 4 else cfg_key

        if typ is int:
            getter = parser.getint
        elif typ is float:
            getter = parser.getfloat
        else:
            getter = parser.get

        if default is None:
            config[dict_key] = getter(section, cfg_key)
        else:
            config[dict_key] = getter(section, cfg_key, fallback=default)

    return config

def load_data(train_path: str, val_path: str) -> tuple[np.ndarray, np.ndarray]:
    train_data = np.memmap(train_path, dtype=np.int32, mode="r")
    val_data = np.memmap(val_path, dtype=np.int32, mode="r")
    return train_data, val_data

def train(
    # Data paths
    train_data: np.ndarray,
    val_data: np.ndarray,
    # Model hyperparameters
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    theta: float,
    eps: float,
    # Optimizer hyperparameters
    lr: float,
    beta1: float,
    beta2: float,
    optimizer_eps: float,
    weight_decay: float,
    # LR scheduler hyperparameters
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    cosine_iters: int,
    # Training hyperparameters
    batch_size: int,
    max_iters: int,
    grad_clip_max_norm: float,
    device: str,
    checkpoint_folder: str = "./checkpoints",
):
    os.makedirs(checkpoint_folder, exist_ok=True)
    metrics_path = os.path.join(checkpoint_folder, "metrics.jsonl")

    model = TransformerLM(d_model, num_heads, d_ff, vocab_size, context_length, num_layers, theta, eps)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=optimizer_eps, weight_decay=weight_decay)

    wall_start = time.time()

    for iter in range(max_iters):
        optimizer.zero_grad()
        x_batch, y_batch = get_batch(train_data, batch_size, context_length, device)
        y_hat = model(x_batch)
        loss = cross_entropy(y_hat, y_batch)
        loss.backward()
        clip_gradient(model.parameters(), grad_clip_max_norm)

        # Update learning rate via cosine schedule before stepping
        new_lr = cosine_lr(iter, lr_max, lr_min, warmup_iters, cosine_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        optimizer.step()

        # Log training and validation performance every EVAL_INTERVAL iterations
        if (iter + 1) % EVAL_INTERVAL == 0:
            train_loss = loss.item()
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(EVAL_ITERS):
                    vx, vy = get_batch(val_data, batch_size, context_length, device)
                    val_losses.append(cross_entropy(model(vx), vy).item())
            model.train()
            avg_val_loss = sum(val_losses) / EVAL_ITERS
            elapsed = time.time() - wall_start
            print(f"iter {iter + 1}: train_loss={train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={new_lr:.6f}, elapsed={elapsed:.1f}s")

            record = {
                "step": iter + 1,
                "train_loss": train_loss,
                "val_loss": avg_val_loss,
                "lr": new_lr,
                "elapsed_seconds": round(elapsed, 2),
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Save checkpoint every CHECKPOINT_INTERVAL iterations
        if (iter + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, iter, os.path.join(checkpoint_folder, f"checkpoint_{iter}.pt"))

def main():
    arg_parser = argparse.ArgumentParser(description="Train a Transformer LM")
    arg_parser.add_argument("--config", type=str, required=True, help="Path to .cfg config file")
    arg_parser.add_argument("--train_data", type=str, required=True, help="Path to training data .npy file")
    arg_parser.add_argument("--val_data", type=str, required=True, help="Path to validation data .npy file")
    arg_parser.add_argument("--checkpoint_folder", type=str, default=None, help="Override checkpoint folder (default: ./checkpoints/<experiment_name>)")
    args = arg_parser.parse_args()

    config = load_config(args.config)

    experiment_name = config.pop("name")
    checkpoint_folder = args.checkpoint_folder or os.path.join("./checkpoints", experiment_name)

    train_data, val_data = load_data(args.train_data, args.val_data)

    train(train_data=train_data, val_data=val_data, checkpoint_folder=checkpoint_folder, **config)


if __name__ == "__main__":
    main()
