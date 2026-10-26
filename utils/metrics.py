# Entropy, Gini, Information Gain calculations

import numpy as np


def entropy(y):
    """
    Рассчитывает энтропию набора меток классов.

    y — список или массив классов, например [0, 0, 1, 1, 1].
    """
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    # Нулевая вероятность не участвует в сумме
    result = -np.sum(probabilities * np.log2(probabilities))
    return 0.0 if result == 0 else float(result)


def gini(y):
    """
    Рассчитывает индекс Джини для набора меток классов.
    """
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    return float(1 - np.sum(probabilities ** 2))


def information_gain(y, y_left, y_right):
    """
    Рассчитывает информационный выигрыш от бинарного разделения.

    y       — метки до разделения;
    y_left  — метки в левой группе;
    y_right — метки в правой группе.
    """
    if len(y) == 0:
        return 0.0

    if len(y_left) + len(y_right) != len(y):
        raise ValueError("Размеры дочерних групп не совпадают с исходной группой")

    parent_entropy = entropy(y)

    left_weight = len(y_left) / len(y)
    right_weight = len(y_right) / len(y)

    children_entropy = (
        left_weight * entropy(y_left)
        + right_weight * entropy(y_right)
    )

    return float(parent_entropy - children_entropy)
