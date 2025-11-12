"""Data preprocessing utilities for cnn_ResNet.
Functions:
- make_datasets: returns (train_ds, val_ds, num_classes)

This module loads the MNIST dataset via tensorflow_datasets and builds small tf.data pipelines.
"""
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess(image, label, img_size=32):
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_datasets(batch_size=128, train_subset=10000, test_subset=1000, img_size=32, seed=42):
    """Load MNIST and return (train_ds, val_ds, num_classes).

    The dataset is shuffled and then .take(train_subset) for quick experiments.
    The function performs dataset loading at call time (so importing this module is cheap).
    """
    (ds_train, ds_val), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    num_classes = ds_info.features['label'].num_classes

    train_ds = ds_train.map(lambda im, lb: preprocess(im, lb, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(10000, seed=seed).take(train_subset).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = ds_val.map(lambda im, lb: preprocess(im, lb, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.take(test_subset).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, num_classes
