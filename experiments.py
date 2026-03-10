import wandb
import numpy as np
from utils.data_loader import load_data

def data_exploration():
    wandb.init(project="mnist-mlp")

    (x_train, y_train), _ = load_data("mnist")
    labels = np.argmax(y_train, axis=1)

    table = wandb.Table(columns=["image","label"])

    for digit in range(10):
        idx = np.where(labels==digit)[0][:5]
        for i in idx:
            img = x_train[i].reshape(28,28)
            table.add_data(wandb.Image(img), digit)

    wandb.log({"sample_images":table})

data_exploration()
