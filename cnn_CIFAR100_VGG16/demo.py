import os
import tensorflow as tf
from tensorflow.data import Dataset

from cnn_CIFAR100_VGG16.data_preprocessing import load_cifar100
from cnn_CIFAR100_VGG16.model import build_fast_vgg, build_vgg16_cifar100
from cnn_CIFAR100_VGG16.train import train_model
from cnn_CIFAR100_VGG16.evaluation import evaluate_and_report

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(RESULTS_DIR, 'vgg_cifar100.keras')

if __name__ == '__main__':
    # smaller subset for quick runs; set to None to use full dataset
    subset_train = 4000
    subset_test = 1000

    (x_train, y_train), (x_test, y_test) = load_cifar100(subset_train=subset_train, subset_test=subset_test, normalize=True)

    # data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    batch_size = 256
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(4000).batch(batch_size).map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # choose model (fast variant by default for speed)
    model = build_fast_vgg()

    # train
    model, history = train_model(model, train_ds, test_ds, MODEL_PATH, epochs=20)

    # evaluate and save reports
    evaluate_and_report(model, test_ds, x_test=x_test, y_test=y_test, results_dir=RESULTS_DIR)

