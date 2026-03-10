import subprocess
import sys

activations = [
    "relu",
    "sigmoid"
]

base_cmd = [
    sys.executable, "src/train.py",
    "-d", "mnist",
    "-e", "15",
    "-b", "64",
    "-l", "cross_entropy",
    "-lr", "0.001",
    "-wd", "0.0",
    "-nhl", "3",
    "-sz", "128", "128", "64",
    "-wi", "xavier",
    "-o", "rmsprop"
]

for act in activations:

    cmd = base_cmd + ["-a", act]

    print("\n====================================")
    print("Running activation function:", act)
    print("Command:", " ".join(cmd))
    print("====================================\n")

    subprocess.run(cmd)

print("\nAll activation function experiments finished.")