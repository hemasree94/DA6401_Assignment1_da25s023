import os
import sys
import json
import argparse
import numpy as np

# Add parent directory to path to allow imports from root level
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import accuracy_score, precision_recall_fscore_support
import wandb

from utils.data_loader import load_data
from mlp.neural_network import NeuralNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST/Fashion-MNIST")

    parser.add_argument("-d", "--dataset",
                        choices=["mnist", "fashion_mnist"],
                        default="mnist")

    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=30)

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=64)

    parser.add_argument("-l", "--loss",
                        choices=["mse", "cross_entropy"],
                        default="mse")

    parser.add_argument("-o", "--optimizer",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        default="rmsprop")

    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default=0.0005)

    parser.add_argument("-wd", "--weight_decay",
                        type=float,
                        default=0.0)

    parser.add_argument("-nhl", "--num_layers",
                        type=int,
                        default=3)

    parser.add_argument("-sz", "--hidden_size",
                        type=int,
                        nargs="+",
                        default=[128, 64, 32])

    # ONE activation used for all hidden layers
    parser.add_argument("-a", "--activation",
                        choices=["relu", "sigmoid", "tanh"],
                        default="sigmoid")

    parser.add_argument("-wi", "--weight_init",
                        choices=["random", "xavier", "zeros"],
                        default="xavier")
    parser.add_argument("-wp", "--wandb_project",
                    type=str,
                    default="mnist-mlp",
                    help="Weights & Biases project name")


    return parser.parse_args()


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, p, r, f1


def save_model(network, path):
    """Save weights and biases of network to numpy file."""
    weights_dict = network.get_weights()
    np.save(path, weights_dict, allow_pickle=True)


def main():
    args = parse_args()

    wandb.init(
        project="minst-mlp",
        config=vars(args)
    )

    # Ensure hidden size list matches num layers
    if len(args.hidden_size) != args.num_layers:
        if len(args.hidden_size) == 1:
            args.hidden_size = args.hidden_size * args.num_layers
        else:
            raise ValueError("Length of --hidden_size must equal --num_layers or be 1")

    # Load data
    (x_train, y_train), (x_test, y_test) = load_data(dataset=args.dataset)

    num_classes = y_train.shape[1]
    input_dim = x_train.shape[1]

    args.input_size = input_dim
    args.output_size = num_classes

    net = NeuralNetwork(args)

    num_samples = x_train.shape[0]
    indices = np.arange(num_samples)

    for epoch in range(1, args.epochs + 1):

        epoch_loss = 0.0
        num_batches = 0

        np.random.shuffle(indices)

        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        for start in range(0, num_samples, args.batch_size):

            end = start + args.batch_size
            xb = x_train_shuffled[start:end]
            yb = y_train_shuffled[start:end]

            # Forward pass
            preds = net.forward(xb)

            # Loss computation
            batch_loss = net.loss.cross_entropy(yb, preds) if args.loss == "cross_entropy" else net.loss.mse(yb, preds)

            epoch_loss += batch_loss
            num_batches += 1

            # Backprop
            net.backward(yb, preds)

            # Update weights
            net.update_weights()

        # Compute metrics
        train_loss = epoch_loss / num_batches

        test_acc, test_prec, test_rec, test_f1 = net.evaluate(x_test, y_test)
        train_acc, _, _, _ = net.evaluate(x_train, y_train)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Train Loss: {train_loss:.4f}"
        )

        # Log every epoch (IMPORTANT)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        })

    print("Training complete.")


if __name__ == "__main__":
    main()
