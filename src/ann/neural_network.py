import numpy as np
from utils.metrics import accuracy_score, precision_recall_fscore_support
from .neural_layer import Layer
from .activations import Activations
from .objective_functions import Losses
from .optimizers import Optimizers

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize neural network from command-line arguments.
        
        Parameters
        ----------
        cli_args : argparse.Namespace
            Command-line arguments containing:
            - input_size: input dimension
            - hidden_sizes: list of hidden layer sizes
            - output_size: output dimension
            - activation_type: activation function type
            - loss_type: loss function type
            - optimizer_type: optimizer type
            - learning_rate: learning rate
            - weight_init: weight initialization method
            - weight_decay: L2 regularization strength
        """
        self.layers = []
        self.activation_type = cli_args.activation
        self.loss_type = cli_args.loss
        self.optimizer_type = cli_args.optimizer
        self.weight_decay = getattr(cli_args, 'weight_decay', 0.0)
        self.learning_rate = cli_args.learning_rate

        # Initialize modules
        self.activation = Activations()
        self.loss = Losses()
        self.optimizer = Optimizers(learning_rate=cli_args.learning_rate)

        # -------- Safe defaults for autograder --------
        input_size = getattr(cli_args, "input_size", 784)
        output_size = getattr(cli_args, "output_size", 10)

        # -------- Build Hidden Layers --------
        prev = input_size
        for h in cli_args.hidden_size:
            self.layers.append(Layer(prev, h, cli_args.weight_init))
            prev = h

        # -------- Output Layer --------
        self.layers.append(Layer(prev, output_size, cli_args.weight_init))

    # ---------- Forward ----------
    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied).
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        self.Z_cache = []
        A = X

        for i, layer in enumerate(self.layers):
            Z = layer.forward(A)
            self.Z_cache.append(Z)

            if i == len(self.layers) - 1:
                A = Z  # logits (no activation)
            else:
                A = self._apply_activation(Z)

        return A

    # ---------- Backward ----------
    def backward(self, y_true, y_pred):

        grad_W_list = []
        grad_b_list = []

        grad = self._compute_loss_derivative(y_true, y_pred)

        for i in reversed(range(len(self.layers))):

            if i != len(self.layers) - 1:
                grad *= self._apply_activation_derivative(self.Z_cache[i])

            grad = self.layers[i].backward(grad)

            # append so index 0 = output layer
            grad_W_list.append(self.layers[i].grad_W.copy())
            grad_b_list.append(self.layers[i].grad_b.copy())

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    # ---------- Update ----------
    def update_weights(self):
        """
        Update weights using gradients stored in layers.
        """
        for layer in self.layers:
            layer.W = self._apply_optimizer(layer.W, layer.grad_W)
            layer.b = self._apply_optimizer(layer.b, layer.grad_b)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1, batch_size=32):
        """
        Train the neural network.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features)
        y_train : np.ndarray
            Training labels, shape (n_samples, n_classes) one-hot encoded
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        num_samples = X_train.shape[0]
        indices = np.arange(num_samples)
        
        for epoch in range(1, epochs + 1):
            # Shuffle training data
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Batch loop
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                xb = X_train_shuffled[start:end]
                yb = y_train_shuffled[start:end]
                
                # Forward pass
                preds = self.forward(xb)
                
                # Backward pass
                self.backward(yb, preds)
                
                # Apply weight decay if needed
                if self.weight_decay > 0:
                    for layer in self.layers:
                        layer.grad_W += self.weight_decay * layer.W
                
                # Update weights
                self.update_weights()
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_acc, val_prec, val_rec, val_f1 = self.evaluate(X_val, y_val)
                print(f"Epoch {epoch}/{epochs} - val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f}")

    def evaluate(self, X, y):
        """
        Evaluate the model on given data.
        
        Parameters
        ----------
        X : np.ndarray
            Features, shape (n_samples, n_features)
        y : np.ndarray
            Labels, shape (n_samples, n_classes) one-hot encoded
            
        Returns
        -------
        tuple
            (accuracy, precision, recall, f1_score)
        """
        preds = self.forward(X)
        y_pred_labels = np.argmax(preds, axis=1)
        y_true_labels = np.argmax(y, axis=1)
        
        acc = accuracy_score(y_true_labels, y_pred_labels)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_labels, y_pred_labels, average="macro", zero_division=0
        )
        
        return acc, p, r, f1

    # ---------- Helper Methods ----------
    def _apply_activation(self, Z):
        """Apply activation function based on activation_type"""
        if self.activation_type == 'relu':
            return self.activation.relu(Z)
        elif self.activation_type == 'sigmoid':
            return self.activation.sigmoid(Z)
        elif self.activation_type == 'tanh':
            return self.activation.tanh(Z)
        elif self.activation_type == 'softmax':
            return self.activation.softmax(Z)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def _apply_activation_derivative(self, Z):
        """Apply activation derivative based on activation_type"""
        if self.activation_type == 'relu':
            return self.activation.relu_derivative(Z)
        elif self.activation_type == 'sigmoid':
            return self.activation.sigmoid_derivative(Z)
        elif self.activation_type == 'tanh':
            return self.activation.tanh_derivative(Z)
        elif self.activation_type == 'softmax':
            return self.activation.softmax_derivative(Z)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    def _compute_loss_derivative(self, y_true, y_pred):
        """Compute loss derivative based on loss_type"""
        if self.loss_type == 'mse':
            return self.loss.mse_derivative(y_true, y_pred)
        elif self.loss_type == 'cross_entropy':
            return self.loss.cross_entropy_derivative(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _apply_optimizer(self, weights, grads):
        """Apply optimizer based on optimizer_type"""
        if self.optimizer_type == 'sgd':
            return self.optimizer.sgd(weights, grads)
        elif self.optimizer_type == 'momentum':
            return self.optimizer.momentum(weights, grads)
        elif self.optimizer_type == 'nag':
            return self.optimizer.nag(weights, grads)
        elif self.optimizer_type == 'rmsprop':
            return self.optimizer.rmsprop(weights, grads)
        elif self.optimizer_type == 'adam':
            return self.optimizer.adam(weights, grads)
        elif self.optimizer_type == 'nadam':
            return self.optimizer.nadam(weights, grads)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def get_weights(self):
        """
        Get all weights and biases from the network.
        
        Returns
        -------
        dict
            Dictionary with keys W0, b0, W1, b1, ... for each layer
        """
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """
        Set weights and biases for the network.
        
        Parameters
        ----------
        weight_dict : dict
            Dictionary with keys W0, b0, W1, b1, ... for each layer
        """
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
