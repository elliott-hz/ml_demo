# CNN MNIST Package

This package contains a simple CNN implementation for MNIST digit classification.

## Package Structure

- `data_processing.py`: Functions for loading and preprocessing the MNIST dataset
- `model.py`: CNN model definition and compilation
- `training.py`: Model training and saving functionality
- `evaluation.py`: Model evaluation and prediction functions
- `demo.py`: Main demo script showing the complete workflow

## Usage

To run the complete CNN training and evaluation workflow:

```bash
python demo.py
```

## Modules

Each module can be imported and used independently:

```python
from cnn_mnist.data_processing import load_and_preprocess_data
from cnn_mnist.model import build_cnn_model
from cnn_mnist.training import train_and_save_model
from cnn_mnist.evaluation import load_model_and_predict, print_classification_report_func
```