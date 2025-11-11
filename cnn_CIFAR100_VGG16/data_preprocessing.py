import numpy as np
from tensorflow.keras import datasets


def load_cifar100(subset_train=None, subset_test=None, normalize=True):
    """
    Load CIFAR-100 and optionally return smaller subsets.

    Args:
        subset_train: int or None, number of training samples to return
        subset_test: int or None, number of test samples to return
        normalize: bool, if True scale pixels to [0,1]

    Returns:
        (x_train, y_train), (x_test, y_test)  where y arrays are 1-D
    """
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

    if subset_train is not None:
        x_train = x_train[:subset_train]
        y_train = y_train[:subset_train]
    if subset_test is not None:
        x_test = x_test[:subset_test]
        y_test = y_test[:subset_test]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    # flatten labels from shape (n,1) to (n,)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return (x_train, y_train), (x_test, y_test)

