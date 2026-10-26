# decision tree implementation

from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    """
    Узел дерева решений.

    Если feature_index равен None, узел является листом.
    """
    prediction: int
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self):
        return self.feature_index is None


import numpy as np

from utils.metrics import entropy


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = None

    def _majority_class(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return int(classes[np.argmax(counts)])

    def _best_split(self, X, y):
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        parent_entropy = entropy(y)
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            values = np.unique(X[:, feature_index])

            # Для числового признака проверяем пороги
            # между соседними уникальными значениями.
            if len(values) < 2:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if not left_mask.any() or not right_mask.any():
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                weighted_entropy = (
                    len(y_left) / n_samples * entropy(y_left)
                    + len(y_right) / n_samples * entropy(y_right)
                )

                gain = parent_entropy - weighted_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        prediction = self._majority_class(y)

        # Условия остановки
        if (
            len(np.unique(y)) == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return Node(prediction=prediction)

        feature, threshold, gain = self._best_split(X, y)

        # Нет полезного разделения
        if feature is None or gain <= 0:
            return Node(prediction=prediction)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_node = self._build_tree(
            X[left_mask], y[left_mask], depth + 1
        )
        right_node = self._build_tree(
            X[right_mask], y[right_mask], depth + 1
        )

        return Node(
            prediction=prediction,
            feature_index=feature,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X должен быть двумерным, а длины X и y должны совпадать")

        if len(y) == 0:
            raise ValueError("Нельзя обучить дерево на пустом наборе данных")

        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.prediction

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)

        return self._predict_one(x, node.right)

    def predict(self, X):
        if self.root is None:
            raise ValueError("Сначала нужно обучить дерево через fit()")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("Количество признаков не совпадает с обучающими данными")

        return np.array([
            self._predict_one(row, self.root)
            for row in X
        ], dtype=int)

    def print_tree(self, feature_names=None):
        if self.root is None:
            print("Дерево ещё не обучено.")
            return

        if feature_names is None:
            feature_names = [
                f"feature_{i}" for i in range(self.n_features)
            ]

        def show(node, indent=""):
            if node.is_leaf:
                print(f"{indent}Predict class = {node.prediction}")
                return

            name = feature_names[node.feature_index]
            print(f"{indent}if {name} <= {node.threshold:.3f}:")
            show(node.left, indent + "  ")
            print(f"{indent}else:")
            show(node.right, indent + "  ")

        show(self.root)
