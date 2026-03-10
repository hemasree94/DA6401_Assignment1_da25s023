import os
import sys
import json
import argparse
import numpy as np

# Add parent directory to path to allow imports from root level
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import accuracy_score, precision_recall_fscore_support

from utils.data_loader import load_data
from mlp.neural_network import NeuralNetwork


def build_config_from_args(args):
    """Build a straight-forward config dict from argparse Namespace so inference can mirror training."""

    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init,
    }

    return config


def make_network_from_config(config, input_size, output_size):
    """Build a NeuralNetwork object from a config dict (similar to what train.py writes)."""

    # Build argparse-like object for NeuralNetwork constructor
    class Args:
        pass

    args = Args()
    for k, v in config.items():
        setattr(args, k, v)

    # Ensure hidden_size is a list
    if isinstance(args.hidden_size, (int, float)):
        args.hidden_size = [int(args.hidden_size)]
    else:
        args.hidden_size = [int(h) for h in args.hidden_size]

    args.input_size = input_size
    args.output_size = output_size

    net = NeuralNetwork(args)

    return net


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def run_inference(args):
    """Run inference on the specified dataset using a saved model (or args-built config)."""

    print(f"\n{'='*70}")
    print(f"Running Inference on {args.dataset.upper()}")
    print(f"{'='*70}\n")

    # Load data
    print(f"Loading {args.dataset} dataset...")
    (_, _), (x_test, y_test) = load_data(dataset=args.dataset)
    print(f"Test set size: {x_test.shape[0]} samples\n")

    # Build config (either from JSON or from the provided args)
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = build_config_from_args(args)

    # Create network and load weights
    print(f"Loading model from {args.weights}...")
    net = make_network_from_config(config, input_size=x_test.shape[1], output_size=y_test.shape[1])
    weights_dict = np.load(args.weights, allow_pickle=True).item()
    net.set_weights(weights_dict)

    print(f"Model configuration: {json.dumps(config, indent=2)}\n")

    # Run inference
    print("Running inference on test set...")
    preds = net.forward(x_test)
    y_pred_labels = np.argmax(preds, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Compute metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average="macro", zero_division=0
    )

    print(f"\n{'='*70}")
    print(f"INFERENCE RESULTS ON {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*70}\n")

    return {
        "dataset": args.dataset,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "y_pred": y_pred_labels.tolist(),
        "y_true": y_true_labels.tolist(),
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a saved neural network model"
    )

    ## Match train.py args so inference can be called with the same CLI style
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", choices=["mse", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128])
    parser.add_argument("-a", "--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("-wi", "--weight_init", choices=["random", "xavier", "zeros"], default="xavier")

    ## Inference-specific args (defaults keep behavior backwards compatible)
    parser.add_argument("-c", "--config", default="best_config.json", help="Optional path to a saved config JSON")
    parser.add_argument("-w", "--weights", default="best_model.npy", help="Path to the saved weights file")
    parser.add_argument("-O", "--output", default="inference_results.json", help="Path to save inference results JSON")

    args = parser.parse_args()

    results = run_inference(args)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
