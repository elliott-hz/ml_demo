"""Demo runner for cnn_ResNet.
Usage (from project root):
python -m cnn_ResNet.demo --epochs 3 --batch_size 128
"""
from tensorflow.keras import callbacks

from data_preprocessing import make_datasets
from evaluation import evaluate_model
from model import build_small_resnet
from train import compile_and_train


def main():
    train_ds, val_ds, num_classes = make_datasets(batch_size=128, train_subset=10000, test_subset=1000)

    model = build_small_resnet(num_classes=num_classes)

    history = compile_and_train(model, train_ds, val_ds, epochs=3, save_path='cnn_resnet.keras')

    report, cm = evaluate_model(model, val_ds)
    print('\nClassification Report:\n', report)
    print('\nConfusion Matrix:\n', cm)


if __name__ == '__main__':
    main()
