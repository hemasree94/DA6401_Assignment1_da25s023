#!/usr/bin/env python3
import random
import json
import subprocess
import sys

N_RUNS = 40   # number of experiments

SEARCH_SPACE = {
    "dataset": ["mnist"],
    "epochs": [30],
    "batch_size": [32, 64],
    "loss": ["cross_entropy", "mse"],
    "optimizer": ["sgd", "rmsprop", "nag", "momentum"],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "weight_decay": [0.0, 0.0001],
    "activation": ["relu", "tanh", "sigmoid"],
    "weight_init": ["random", "xavier"],
    "hidden_size": [
        [128],
        [128, 64],
        [128, 64, 32],
        [128, 64, 32, 16],
    ]
}


def sample_config():
    config = {}
    for k, v in SEARCH_SPACE.items():
        config[k] = random.choice(v)
    return config


def run_training(config):

    cmd = [
        sys.executable, "src/train.py",
        "-d", config["dataset"],
        "-e", str(config["epochs"]),
        "-b", str(config["batch_size"]),
        "-l", config["loss"],
        "-o", config["optimizer"],
        "-lr", str(config["learning_rate"]),
        "-wd", str(config["weight_decay"]),
        "-a", config["activation"],
        "-wi", config["weight_init"],
        "-nhl", str(len(config["hidden_size"])),
        "-sz"
    ] + [str(s) for s in config["hidden_size"]]

    print("\nRunning:", " ".join(cmd))

    result = subprocess.run(cmd)

    return result.returncode == 0


def main():

    best_f1 = -1

    for i in range(N_RUNS):

        config = sample_config()

        print(f"\nRun {i+1}/{N_RUNS}")

        success = run_training(config)

        if success:
            try:
                with open("best_config.json") as f:
                    cfg = json.load(f)

                f1 = cfg.get("best_f1", -1)

                if f1 > best_f1:
                        best_f1 = f1
                        print("New best F1:", best_f1)

                        # Save global best config
                        with open("global_best_config.json", "w") as f:
                            json.dump(cfg, f, indent=2)

                        # Save global best model
                        import shutil
                        shutil.copy("best_model.npy", "global_best_model.npy")

            except:
                pass

    print("\nSearch finished")
    print("Best F1:", best_f1)


if __name__ == "__main__":
    main()