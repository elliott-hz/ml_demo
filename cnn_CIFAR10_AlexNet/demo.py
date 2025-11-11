import os
from tensorflow.data import Dataset

from cnn_CIFAR10_AlexNet.data_preprocessing import load_cifar10
from cnn_CIFAR10_AlexNet.model import SmallAlexNet
from cnn_CIFAR10_AlexNet.train import train
from cnn_CIFAR10_AlexNet.evaluation import evaluate_and_report


RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

if __name__ == '__main__':
    # small subset for quick run
    (x_train, y_train), (x_test, y_test) = load_cifar10(subset_train=10240, subset_test=1024, normalize=True)

    # build dataset pipeline with augmentation
    import tensorflow as tf
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(128).map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128).prefetch(tf.data.AUTOTUNE)

    model = SmallAlexNet()
    model, history = train(model, train_ds, val_ds, epochs=20, batch_size=128, model_path=os.path.join(RESULTS_DIR, 'small_alexnet.h5'))

    evaluate_and_report(model, x_test, y_test, results_dir=RESULTS_DIR)

