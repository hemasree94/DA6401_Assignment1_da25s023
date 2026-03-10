import os
import sys
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.data_loader import load_data
from mlp.neural_network import NeuralNetwork


def load_model(config_path="best_config.json", weights_path="best_model.npy"):
    """
    Load the best trained model from saved config and weights.
    
    Parameters
    ----------
    config_path : str
        Path to the best configuration JSON file
    weights_path : str
        Path to the best model weights NPY file
        
    Returns
    -------
    NeuralNetwork
        Reconstructed neural network with loaded weights
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create a simple namespace object to mimic argparse.Namespace
    class Config:
        pass
    
    args = Config()
    for key, value in config.items():
        setattr(args, key, value)
    
    # Handle hidden_size properly - ensure it's a list
    if isinstance(args.hidden_size, (int, float)):
        args.hidden_size = [int(args.hidden_size)]
    else:
        args.hidden_size = [int(h) for h in args.hidden_size]
    
    # Create network
    net = NeuralNetwork(args)
    
    # Load weights
    weights_dict = np.load(weights_path, allow_pickle=True).item()
    net.set_weights(weights_dict)
    
    return net, config


def run_inference(dataset="mnist", config_path="best_config.json", weights_path="best_model.npy"):
    """
    Run inference on test set and report metrics.
    
    Parameters
    ----------
    dataset : str
        Dataset to use ('mnist' or 'fashion_mnist')
    config_path : str
        Path to the best configuration JSON file
    weights_path : str
        Path to the best model weights NPY file
    """
    print(f"\n{'='*70}")
    print(f"Running Inference on {dataset.upper()}")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"Loading {dataset} dataset...")
    (_, _), (x_test, y_test) = load_data(dataset=dataset)
    print(f"Test set size: {x_test.shape[0]} samples\n")
    
    # Load model
    print(f"Loading model from {weights_path}...")
    net, config = load_model(config_path, weights_path)
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
    print(f"INFERENCE RESULTS ON {dataset.upper()}")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*70}\n")
    
    return {
        "dataset": dataset,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "y_pred": y_pred_labels.tolist(),
        "y_true": y_true_labels.tolist(),
        "config": config
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a saved neural network model"
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=["mnist", "fashion_mnist"],
        default="mnist",
        help="Dataset to test on"
    )
    parser.add_argument(
        "-c", "--config",
        default="best_config.json",
        help="Path to best configuration JSON file"
    )
    parser.add_argument(
        "-w", "--weights",
        default="best_model.npy",
        help="Path to best model weights NPY file"
    )
    parser.add_argument(
        "-o", "--output",
        default="inference_results.json",
        help="Path to save inference results JSON"
    )
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        dataset=args.dataset,
        config_path=args.config,
        weights_path=args.weights
    )
    
    # Save results to JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}\n")


if __name__ == "__main__":
    main()
