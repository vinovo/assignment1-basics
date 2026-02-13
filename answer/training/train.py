import argparse
import configparser


# Schema: (section, key, type, default or None if required)
CONFIG_SCHEMA = [
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
    ("optimizer", "eps", float, 1e-8),
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
    ("training", "checkpoint_interval", int, 1000),
    ("training", "eval_interval", int, 100),
    ("training", "eval_iters", int, 10),
    ("training", "device", str, "cuda"),
]


def load_config(path: str) -> dict:
    """Load a .cfg file and return a flat dict of all hyperparameters."""
    parser = configparser.ConfigParser()
    parser.read(path)

    config = {}
    for section, key, typ, default in CONFIG_SCHEMA:
        if typ is int:
            getter = parser.getint
        elif typ is float:
            getter = parser.getfloat
        else:
            getter = parser.get

        if default is None:
            # Required field — will raise if missing
            config[key] = getter(section, key)
        else:
            config[key] = getter(section, key, fallback=default)

    return config


def train(
    # Data paths
    train_path: str,
    val_path: str,
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
    checkpoint_interval: int,
    eval_interval: int,
    eval_iters: int,
    device: str,
):
    # TODO: implement training loop
    pass


def main():
    arg_parser = argparse.ArgumentParser(description="Train a Transformer LM")
    arg_parser.add_argument("--config", type=str, required=True, help="Path to .cfg config file")
    arg_parser.add_argument("--train_data", type=str, required=True, help="Path to training data .npy file")
    arg_parser.add_argument("--val_data", type=str, required=True, help="Path to validation data .npy file")
    args = arg_parser.parse_args()

    config = load_config(args.config)

    # Rename to avoid collision with model eps
    config["optimizer_eps"] = config.pop("eps")

    train(train_path=args.train_data, val_path=args.val_data, **config)


if __name__ == "__main__":
    main()
