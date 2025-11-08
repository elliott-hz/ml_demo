# ==============================================
# Step 1.1: Loading MNIST Dataset
# ==============================================
import os
import pickle

# noinspection PyUnresolvedReferences
from tensorflow.keras.datasets import mnist
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import to_categorical

# Define the 7 dataset size levels (training size, testing size)
# Testing size is 15% of training size
DATASET_LEVELS = {
    'level_1': (1024, 1000),
    'level_2': (2048, 1000),
    'level_3': (4096, 1000),
    'level_4': (8192, 2000),
    'level_5': (16384, 2000),
    'level_6': (32768, 5000),
    'level_7': (60000, 5000)  # Full MNIST training set
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


def load_dataset(train_size=None, test_size=None, level=None, augment=False):
    """
    Load MNIST dataset either from remote source or from local pre-processed files
    
    Parameters:
        train_size (int): Number of training samples (used when level is None)
        test_size (int): Number of test samples
        level (str): Pre-defined dataset level ('level_1' to 'level_7')
        augment (bool): Whether to apply data augmentation
        
    Returns:
        tuple: Prepared dataset (x_train, y_train, x_test, y_test)
    """
    # Determine sizes
    if level is not None:
        if level not in DATASET_LEVELS:
            raise ValueError(f"Invalid level. Choose from: {list(DATASET_LEVELS.keys())}")
        train_size, test_size = DATASET_LEVELS[level]

    # Try to load from local pre-processed files
    if level is not None:
        filename = os.path.join(RESULTS_DIR, f"mnist_{level}.pkl")
        if os.path.exists(filename):
            print(f"Loading pre-processed {level} dataset from local file...")
            with open(filename, 'rb') as f:
                x_train, y_train, x_test, y_test = pickle.load(f)
        else:
            print(f"Pre-processed {level} dataset not found. Creating it now...")
            # Load from remote source
            (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()

            # Select subset of training data
            x_train = x_train_full[:train_size]
            y_train = y_train_full[:train_size]

            # Select subset of test data
            x_test = x_test_full[:test_size]
            y_test = y_test_full[:test_size]

            # Save pre-processed dataset for future use
            with open(filename, 'wb') as f:
                pickle.dump((x_train, y_train, x_test, y_test), f)
            print(f"Saved {level} dataset to {filename}")
    else:
        print(f"Loading MNIST dataset from remote source...")
        (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()

        # Select subset of training data
        x_train = x_train_full[:train_size] if train_size else x_train_full
        y_train = y_train_full[:train_size] if train_size else y_train_full

        # Select subset of test data
        x_test = x_test_full[:test_size] if test_size else x_test_full
        y_test = y_test_full[:test_size] if test_size else y_test_full

    print(f"Data loading completed: ")
    print("-" * 50)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print("-" * 50)

    # Apply image preprocessing
    print(f"\n\nImage preprocessing: ")
    print("-" * 50)
    # Reshape to 4D tensor format (N, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    print(f"-- x_train preprocessing completed: {x_train.shape}")
    print(f"-- x_test preprocessing completed: {x_test.shape}")
    print("-" * 50)

    # Apply one-hot encoding
    print(f"\n\nOne-hot encoding: ")
    print("-" * 50)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    print(f"y_train encoded shape: {y_train.shape}")
    print(f"y_test encoded shape: {y_test.shape}")
    print("-" * 50)

    # Apply data augmentation if requested
    if augment:
        print(f"\n\nApplying data augmentation: ")
        print("-" * 50)
        datagen = ImageDataGenerator(
            rotation_range=10,  # +/- 10 degree rotation
            width_shift_range=0.1,  # +/- 10% horizontal shift
            height_shift_range=0.1,  # +/- 10% vertical shift
            zoom_range=0.1  # 10% zoom in/out
        )
        datagen.fit(x_train)
        print("Data augmentation setup completed.")
        print("-" * 50)

    return x_train, y_train, x_test, y_test
