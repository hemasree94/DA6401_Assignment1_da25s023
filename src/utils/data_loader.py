import os
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist


def _one_hot(labels, num_classes=10):
    """Convert integer labels to one-hot vectors."""
    return np.eye(num_classes)[labels]


def _normalize_images(x):
    # scale pixels to [0,1] and flatten
    x = x.astype(np.float32) / 255.0
    return x.reshape(x.shape[0], -1)


def download_and_save(dataset='mnist', data_dir='data'):
    """Download specified dataset and save numpy arrays to disk.

    Parameters
    ----------
    dataset : str
        'mnist' or 'fashion_mnist'.
    data_dir : str
        Directory where .npy files will be stored.

    Returns
    -------
    tuple
        ((x_train, y_train), (x_test, y_test))
    """
    os.makedirs(data_dir, exist_ok=True)

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    x_train = _normalize_images(x_train)
    x_test = _normalize_images(x_test)
    y_train = _one_hot(y_train)
    y_test = _one_hot(y_test)

    np.save(os.path.join(data_dir, f"{dataset}_x_train.npy"), x_train)
    np.save(os.path.join(data_dir, f"{dataset}_y_train.npy"), y_train)
    np.save(os.path.join(data_dir, f"{dataset}_x_test.npy"), x_test)
    np.save(os.path.join(data_dir, f"{dataset}_y_test.npy"), y_test)

    return (x_train, y_train), (x_test, y_test)


def load_data(dataset='mnist', data_dir='data'):
    """Load dataset from disk, downloading if necessary."""
    paths = {
        'x_train': os.path.join(data_dir, f"{dataset}_x_train.npy"),
        'y_train': os.path.join(data_dir, f"{dataset}_y_train.npy"),
        'x_test': os.path.join(data_dir, f"{dataset}_x_test.npy"),
        'y_test': os.path.join(data_dir, f"{dataset}_y_test.npy"),
    }

    if not all(os.path.exists(p) for p in paths.values()):
        return download_and_save(dataset, data_dir)

    x_train = np.load(paths['x_train'])
    y_train = np.load(paths['y_train'])
    x_test = np.load(paths['x_test'])
    y_test = np.load(paths['y_test'])

    return (x_train, y_train), (x_test, y_test)