"""Sigmoid function"""

import numpy as np


def sigmoid(x):
    """Applies sigmoid function to NumPy matrix"""

    # return 1 / (1 + np.exp(-matrix))
    return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
