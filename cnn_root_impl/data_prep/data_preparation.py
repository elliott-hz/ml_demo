# ==============================================
# Step 1.1: Loading MNIST Dataset
# ==============================================
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from tensorflow.keras.datasets import mnist
# noinspection PyUnresolvedReferences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# noinspection PyUnresolvedReferences
from tensorflow.keras.utils import to_categorical


def loading_MNIST_Dataset(train_size=16, val_size=16, test_size=10):
    """
    Load and split MNIST dataset into train, validation and test sets
    
    Parameters:
        train_size (int): Number of training samples
        val_size (int): Number of validation samples
        test_size (int): Number of test samples
        
    Returns:
        tuple: Split datasets (x_train, y_train, x_val, y_val, x_test, y_test)
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=train_size, test_size=val_size, random_state=42
    )

    # Use only a subset of the test dataset
    x_test, y_test = x_test[:test_size], y_test[:test_size]
    print(f"\n\nData loading completed: ")
    print("-" * 50)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print("-" * 50)
    return x_train, y_train, x_val, y_val, x_test, y_test


def show_image(x_train, y_train):
    """
    Display sample images from the training set
    
    Parameters:
        x_train (np.array): Training images
        y_train (np.array): Training labels
    """
    print(f"\n\nShowing sample images: ")
    print("-" * 50)
    # Visualization of first 5 images from the training set
    plt.figure(figsize=(10, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.suptitle("MNIST Training Samples")
    plt.show()
    print("-" * 50)


# ==============================================
# Step 1.2: Image Preprocessing
# ==============================================
def _image_preprocess_single(img):
    """
    Preprocess a single image using morphological operations
    
    Steps:
    1. Convert to uint8 format for OpenCV morphological operations
    2. Apply opening operation to remove exterior noise
    3. Apply closing operation to fill interior gaps
    4. Normalize pixel values to [0,1] range
    
    Parameters:
        img (np.array): Input image
        
    Returns:
        np.array: Preprocessed image
    """
    # Ensure uint8 format (1 gray level channel) for OpenCV morphological operations
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8)  # Rescale to [0, 255] if original scale is [0,1] float
    else:
        img_uint8 = img.copy().astype(np.uint8)  # Copy directly if already in [0, 255] range

    # Define structuring element (kernel) of size 2x2
    # Small kernel affects only small noises while preserving the main structure
    kernel = np.ones((2, 2), np.uint8)

    # Opening operation (erosion followed by dilation): removes small exterior noises
    img_open = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)

    # Closing operation (dilation followed by erosion): fills small interior holes and smooths boundaries
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)

    # Convert from uint8 to float32 and normalize to [0,1] range
    # Normalization helps achieve more stable convergence during neural network training
    # Values in [0,255] range may cause computational overflow in weighted sums
    img_normalized = img_close.astype('float32') / 255.0

    return img_normalized


def image_preprocess(dataset):
    """
    Apply image preprocessing to the entire dataset
    
    Parameters:
        dataset (tuple): Dataset tuple (x_train, y_train, x_val, y_val, x_test, y_test)
        
    Returns:
        tuple: Preprocessed dataset with images reshaped to 4D tensors
    """
    print(f"\n\nImage preprocessing: ")
    print("-" * 50)
    (x_train, y_train, x_val, y_val, x_test, y_test) = dataset

    # Process each dataset split and reshape to 4D tensor format (N, 28, 28, 1)
    # This ensures the data is in the proper format required by CNNs (NHWC format)
    x_train = np.array([_image_preprocess_single(img) for img in x_train]).reshape(-1, 28, 28, 1)
    print(f"-- x_train preprocessing completed: {len(x_train)} images with shape: {x_train.shape}")
    x_val = np.array([_image_preprocess_single(img) for img in x_val]).reshape(-1, 28, 28, 1)
    print(f"-- x_val preprocessing completed: {len(x_val)} images with shape: {x_val.shape}")
    x_test = np.array([_image_preprocess_single(img) for img in x_test]).reshape(-1, 28, 28, 1)
    print(f"-- x_test preprocessing completed: {len(x_test)} images with shape: {x_test.shape}")
    print("-" * 50)

    return x_train, y_train, x_val, y_val, x_test, y_test


# ==============================================
# Step 1.3: One-hot Encoding of Labels
# ==============================================
def apply_one_hot(dataset):
    """
    Apply one-hot encoding to dataset labels
    
    Parameters:
        dataset (tuple): Dataset tuple (x_train, y_train, x_val, y_val, x_test, y_test)
        
    Returns:
        tuple: Dataset with one-hot encoded labels
    """
    (x_train, y_train, x_val, y_val, x_test, y_test) = dataset

    # Apply one-hot encoding to all dataset splits
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    print(f"\n\nOne-hot encoding completed: ")
    print("-" * 50)
    print(f"y_train encoded shape: {y_train.shape}")
    print(f"y_val encoded shape: {y_val.shape}")
    print(f"y_test encoded shape: {y_test.shape}")
    print("-" * 50)

    return x_train, y_train, x_val, y_val, x_test, y_test


# ==============================================
# Step 1.4: Data Augmentation
# ==============================================
def augment_img(dataset):
    """
    Set up data augmentation for the training dataset
    
    Parameters:
        dataset (tuple): Dataset tuple (x_train, y_train, x_val, y_val, x_test, y_test)
        
    Returns:
        tuple: Dataset without augmentation generator (same as input)
    """
    (x_train, y_train, x_val, y_val, x_test, y_test) = dataset

    # MNIST contains various handwritten digit styles.
    # Data augmentation improves model robustness during training.

    # Using relatively mild augmentation parameters (10% shift, 10 degree rotation)
    # to avoid excessive distortion that might change digit semantics.
    # For example, with extreme parameters, digit '6' might be mistaken for '9'.
    datagen = ImageDataGenerator(
        rotation_range=10,  # +/- 10 degree rotation
        width_shift_range=0.1,  # +/- 10% horizontal shift
        height_shift_range=0.1,  # +/- 10% vertical shift
        zoom_range=0.1  # 10% zoom in/out
    )
    datagen.fit(x_train)
    print(f"\n\nImage Augmentation on x_train completed.")
    print("-" * 50)
    # Return dataset without datagen as it's not used in subsequent steps
    return x_train, y_train, x_val, y_val, x_test, y_test


# ==============================================
# Step 1.5: Data Loading and Preprocessing Pipeline
# ==============================================
def print_step_head(head_name, _index=-1):
    """
    Print formatted section header
    
    Parameters:
        head_name (str): Header text
        _index (int): Step number
    """
    print("\n\n")
    print("=" * 50)
    print(f"{(str(_index) + '. ') if _index > -1 else ''}{head_name}")
    print("=" * 50)


def load_and_prepare_data(train_size=16, val_size=8, test_size=10,
                          show_demo=True, preprocess=True,
                          one_hot=True, augment=True):
    """
    Main function to load and prepare MNIST data
    
    Parameters:
        train_size (int): Number of training samples
        val_size (int): Number of validation samples
        test_size (int): Number of test samples
        show_demo (bool): Whether to display sample images
        preprocess (bool): Whether to apply image preprocessing
        one_hot (bool): Whether to apply one-hot encoding
        augment (bool): Whether to set up data augmentation
        
    Returns:
        tuple: Prepared dataset (x_train, y_train, x_val, y_val, x_test, y_test)
    """

    print_step_head("load_and_prepare_data", _index=1)
    dataset_original = loading_MNIST_Dataset(train_size, val_size, test_size)
    if show_demo:
        show_image(dataset_original[0], dataset_original[1])
    if preprocess:
        dataset = image_preprocess(dataset_original)
    else:
        dataset = dataset_original
    if one_hot:
        dataset = apply_one_hot(dataset)
    if augment:
        dataset = augment_img(dataset)
    return dataset
