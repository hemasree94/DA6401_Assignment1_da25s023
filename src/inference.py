import os
import sys
import json
import argparse
import numpy as np

# allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import accuracy_score, precision_recall_fscore_support
from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained neural network"
    )

    # same arguments as train.py
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", choices=["mse", "cross_entropy"], default="mse")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="rmsprop")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0005)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("-a", "--activation", choices=["relu", "sigmoid", "tanh"], default="sigmoid")
    parser.add_argument("-wi", "--weight_init", choices=["random", "xavier", "zeros"], default="xavier")

    # inference specific
    parser.add_argument("-c", "--config", default="src/best_config.json")
    parser.add_argument("-w", "--weights", default="src/best_model.npy")
    parser.add_argument("-O", "--output", default="src/inference_results.json")

    return parser.parse_args()


def build_config_from_args(args):
    return {
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


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def make_network(config, input_size, output_size):

    class Args:
        pass

    args = Args()

    for k, v in config.items():
        setattr(args, k, v)

    args.input_size = input_size
    args.output_size = output_size

    if isinstance(args.hidden_size, int):
        args.hidden_size = [args.hidden_size]

    return NeuralNetwork(args)


def run_inference(args):

    print("\n==============================")
    print("Running Inference")
    print("==============================\n")

    (_, _), (x_test, y_test) = load_data(dataset=args.dataset)

    print(f"Test samples: {x_test.shape[0]}")

    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = build_config_from_args(args)

    net = make_network(
        config,
        input_size=x_test.shape[1],
        output_size=y_test.shape[1],
    )

    weights = np.load(args.weights, allow_pickle=True).item()
    net.set_weights(weights)

    preds = net.forward(x_test)

    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print("\nResults")
    print("-------------------")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    results = {
        "dataset": args.dataset,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "y_pred": y_pred.tolist(),
        "y_true": y_true.tolist(),
        "config": config,
    }

    return results


def main():

    args = parse_arguments()

    results = run_inference(args)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()