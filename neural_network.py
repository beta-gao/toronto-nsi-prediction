from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class NeuralNetworkRegressor:
    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64),
        learning_rate: float = 0.001,
        epochs: int = 250,
        batch_size: int = 256,
        patience: int = 20,
        l2_lambda: float = 1e-4,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        self.verbose = verbose

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.history_ = TrainingHistory(train_loss=[], val_loss=[])

    def _initialize_parameters(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.random_state)
        layer_sizes = [input_dim, *self.hidden_layers, 1]

        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(2.0 / in_dim)
            self.weights.append(rng.normal(0.0, limit, size=(in_dim, out_dim)))
            self.biases.append(np.zeros((1, out_dim)))

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.feature_mean_ = X.mean(axis=0, keepdims=True)
        self.feature_std_ = X.std(axis=0, keepdims=True)
        self.feature_std_[self.feature_std_ == 0] = 1.0
        return (X - self.feature_mean_) / self.feature_std_

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError("Model must be fitted before calling predict.")
        return (X - self.feature_mean_) / self.feature_std_

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations = []

        current = X
        for layer_idx, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weights + bias
            pre_activations.append(z)
            if layer_idx == len(self.weights) - 1:
                current = z
            else:
                current = self._relu(z)
            activations.append(current)
        return activations, pre_activations

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        l2_penalty = sum(np.sum(weights ** 2) for weights in self.weights)
        return float(mse + self.l2_lambda * l2_penalty / y_true.shape[0])

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        batch_size = y_true.shape[0]
        grad_w = [np.zeros_like(weights) for weights in self.weights]
        grad_b = [np.zeros_like(bias) for bias in self.biases]

        delta = 2.0 * (activations[-1] - y_true) / batch_size
        for layer_idx in reversed(range(len(self.weights))):
            grad_w[layer_idx] = (
                activations[layer_idx].T @ delta
                + (2.0 * self.l2_lambda / batch_size) * self.weights[layer_idx]
            )
            grad_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx > 0:
                delta = (delta @ self.weights[layer_idx].T) * self._relu_grad(
                    pre_activations[layer_idx - 1]
                )
        return grad_w, grad_b

    def fit(self, X, y, X_val=None, y_val=None):
        self.history_ = TrainingHistory(train_loss=[], val_loss=[])
        X_train = np.asarray(X, dtype=float)
        y_train = np.asarray(y, dtype=float).reshape(-1, 1)
        X_train = self._standardize_fit(X_train)

        X_val_arr = None
        y_val_arr = None
        if X_val is not None and y_val is not None:
            X_val_arr = self._standardize_transform(np.asarray(X_val, dtype=float))
            y_val_arr = np.asarray(y_val, dtype=float).reshape(-1, 1)

        self._initialize_parameters(X_train.shape[1])

        best_state = None
        best_val_loss = float("inf")
        stale_epochs = 0
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            indices = rng.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, X_shuffled.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                activations, pre_activations = self._forward(X_batch)
                grad_w, grad_b = self._backward(activations, pre_activations, y_batch)

                for layer_idx in range(len(self.weights)):
                    self.weights[layer_idx] -= self.learning_rate * grad_w[layer_idx]
                    self.biases[layer_idx] -= self.learning_rate * grad_b[layer_idx]

            train_pred = self._forward(X_train)[0][-1]
            train_loss = self._compute_loss(y_train, train_pred)
            self.history_.train_loss.append(train_loss)

            if X_val_arr is not None and y_val_arr is not None:
                val_pred = self._forward(X_val_arr)[0][-1]
                val_loss = self._compute_loss(y_val_arr, val_pred)
                self.history_.val_loss.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    stale_epochs = 0
                    best_state = (
                        copy.deepcopy(self.weights),
                        copy.deepcopy(self.biases),
                    )
                else:
                    stale_epochs += 1

                if self.verbose and (epoch + 1 == 1 or (epoch + 1) % 25 == 0):
                    print(
                        f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} "
                        f"| val_loss={val_loss:.6f}"
                    )

                if stale_epochs >= self.patience:
                    if best_state is not None:
                        self.weights, self.biases = best_state
                    break
            elif self.verbose and (epoch + 1 == 1 or (epoch + 1) % 25 == 0):
                print(f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f}")

        if best_state is not None:
            self.weights, self.biases = best_state

        return self

    def predict(self, X) -> np.ndarray:
        X_arr = self._standardize_transform(np.asarray(X, dtype=float))
        predictions = self._forward(X_arr)[0][-1]
        return predictions.ravel()
