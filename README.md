# DA6401 Assignment 1 (MLP on MNIST / Fashion-MNIST)

This project implements a fully-connected neural network (MLP) from scratch using NumPy, trained on MNIST dataset.

This repository provides:

- An **MLP implementation** with configurable architecture, activations, and optimizers.
- A **training script** with command-line arguments and Weights & Biases (wandb) logging.
- An **random search helper** to explore hyperparameters and keep the best model.
- An **inference script** to evaluate saved models and export results.
- Automatic dataset download + caching using NumPy `.npy` files.

---


### 1) Setup 

Install dependencies from requirements.txt

### 2) Train the model

```bash
python src/train.py -d mnist -e 30 -b 64 -l cross_entropy -o rmsprop -lr 0.0005 -nhl 3 -sz 128 64 32 -a sigmoid -wi xavier
```

The script will save a model checkpoint to `best_model.npy` and the training configuration to `best_config.json`.

### 3) Run inference (evaluate saved model)

You can run inference using the same argument style as training. If no config file exists, the CLI arguments are used to build the model.

```bash
python src/inference.py \
  -d mnist \
  -e 30 \
  -b 64 \
  -l cross_entropy \
  -o rmsprop \
  -lr 0.0005 \
  -nhl 3 \
  -sz 128 64 32 \
  -a sigmoid \
  -wi xavier
```

By default it will load weights from `best_model.npy` and save results to `inference_results.json`.

---

### Scripts

- `src/train.py` — Training loop with logging and metric tracking.
- `src/inference.py` — Load a saved config + weights and evaluate on the test set.
- `hyperparameter_search.py` — Run a random hyperparameter search and keep the best model.

### Code modules

- `src/mlp/` — Neural network implementation (layers, activations, optimizers, loss).
- `src/utils/data_loader.py` — Downloads MNIST/Fashion-MNIST and caches `.npy` files.

### Data

- `data/` — Cached `.npy` datasets (auto-generated on first run).

---

## Notes
- Uses `wandb` for logging. 
- Datasets are downloaded automatically in the first time  run `src/train.py` or `src/inference.py`.

---


## 🗂️ Project structure

```
.
├── src
│   ├── train.py
│   ├── inference.py
│   ├── mlp/        # network implementation
│   └── utils/      # data loading + helpers
├── data/           # cached datasets (npys)
├── best_model.npy
├── best_config.json
├── global_best_model.npy
├── global_best_config.json
├── grid_search.py
└── requirements.txt
```

## Wandb project report link
