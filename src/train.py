import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

# Add parent directory to path to allow imports from root level
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_data
from mlp.neural_network import NeuralNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST/Fashion-MNIST")

    parser.add_argument("-d", "--dataset",
                        choices=["mnist", "fashion_mnist"],
                        default="mnist")

    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=10)

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=64)

    parser.add_argument("-l", "--loss",
                        choices=["mse", "cross_entropy"],
                        default="cross_entropy")

    parser.add_argument("-o", "--optimizer",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        default="sgd")

    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default=0.01)

    parser.add_argument("-wd", "--weight_decay",
                        type=float,
                        default=0.0)

    parser.add_argument("-nhl", "--num_layers",
                        type=int,
                        default=1)

    parser.add_argument("-sz", "--hidden_size",
                        type=int,
                        nargs="+",
                        default=[128])

    # ONE activation used for all hidden layers
    parser.add_argument("-a", "--activation",
                        choices=["relu", "sigmoid", "tanh"],
                        default="relu")

    parser.add_argument("-wi", "--weight_init",
                        choices=["random", "xavier"],
                        default="xavier")

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
            project="mnist-mlp",
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

    # Add input and output dimensions to args
    args.input_size = input_dim
    args.output_size = num_classes

    # Create network with cli_args
    net = NeuralNetwork(args)

    # Train the network
    best_f1 = -1.0
    best_config = None
    best_weights_path = "best_model.npy"

    num_samples = x_train.shape[0]
    indices = np.arange(num_samples)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        # Shuffle training data
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]

        # Batch loop
        for start in range(0, num_samples, args.batch_size):
            end = start + args.batch_size
            xb = x_train_shuffled[start:end]
            yb = y_train_shuffled[start:end]

            # Forward pass
            # Forward pass
            preds = net.forward(xb)

            # Compute training loss
            batch_loss = net.loss.cross_entropy(yb, preds) if args.loss == "cross_entropy" else net.loss.mse(yb, preds)

            epoch_loss += batch_loss
            num_batches += 1

            # Backward pass
            net.backward(yb, preds)
            
            # Apply weight decay if needed
            if args.weight_decay > 0:
                for layer in net.layers:
                    layer.grad_W += args.weight_decay * layer.W

            # Update weights
            net.update_weights()

        # Evaluate on test set
        grad_norm = np.linalg.norm(net.layers[0].grad_W)
        train_loss = epoch_loss / num_batches
        test_acc, test_prec, test_rec, test_f1 = net.evaluate(x_test, y_test)
        
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Accuracy: {test_acc:.4f} | "
            f"Precision: {test_prec:.4f} | "
            f"Recall: {test_rec:.4f} | "
            f"F1-score: {test_f1:.4f}"
            f" | Train Loss: {train_loss:.4f}"
        )

        wandb.log({
            "grad_norm": grad_norm,
            "train_loss": train_loss,
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
        })

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_config = vars(args).copy()
            weights_dict = net.get_weights()
            np.save(best_weights_path, weights_dict, allow_pickle=True)
    

    # Save config with best F1 score
    if best_config is not None:
        best_config["best_f1"] = float(best_f1)
    with open("best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    

    print(f"Training complete. Best F1 {best_f1:.4f}, weights saved to {best_weights_path}")


if __name__ == "__main__":
    main()
