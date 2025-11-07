# -------------------------------
# Step 1.1: Loading MNIST DataSet
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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

    # Subset of dataset for testing
    x_test, y_test = x_test[:test_size], y_test[:test_size]
    print(f"\n\nData loading completed: ")
    print("-" * 50)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    print("-" * 50)
    return (x_train, y_train, x_val, y_val, x_test, y_test)

def show_image(x_train, y_train):
    """
    Display sample images from the training set
    
    Parameters:
        x_train (np.array): Training images
        y_train (np.array): Training labels
    """
    print(f"\n\nShow image: ")
    print("-" * 50)
    # visualization of top 5 number images
    plt.figure(figsize=(10,4))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.suptitle("MNIST Training Samples")
    plt.show()
    print("-" * 50)

# -------------------------------
# Step 1.2: Image Preprocessing
# -------------------------------
def _image_preprocess_single(img):
    """
    Image preprocessing:
    - uint8 transformation: for Binary Morphological Operations (opening / closing etc.)
    - Opening: Removes exterior noise
    - Closing: Fills interior gaps
    - Normalisation: scaling to [0,1]
    """
    # ensure uint8 format(1 graylevel channel) for the binary morphological operations (opening and closing) in OpenCV
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8)  # rescaling to [0, 255] if the original scale is [0,1] float
    else:
        img_uint8 = img.copy().astype(np.uint8)   # directly copy for operations if it already is [0, 255]

    # structuring element (kernel) size is (2x2)
    # -> small kernel only impact small noises, avoiding destroy the main structure of the original image
    kernel = np.ones((2,2), np.uint8)

    # Opening operation (erosion first dilation then): eliminate small exterior noises
    img_open = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)

    # Closing operation (dilation->erosion): fills small interior holds, connects nearby objects and smooths boundaries
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)

    # convert from unit8 to float32 and normalize to [0,1]
    # Why: to enable more stable convergence when training NN
    # What if [0,255]: numbers of weighted sum probably out of the boundaries of limitation in computation
    img_normalized = img_close.astype('float32') / 255.0

    return img_normalized

def image_preprocess(dataset):
    """
    Apply image preprocessing to entire dataset
    
    Parameters:
        dataset (tuple): Dataset tuple (x_train, y_train, x_val, y_val, x_test, y_test)
        
    Returns:
        tuple: Preprocessed dataset
    """
    print(f"\n\nImage preprocessing: ")
    print("-" * 50)
    (x_train, y_train, x_val, y_val, x_test, y_test) = dataset

    # Batch dimension assign to -1 for automatically inference of the number of N (number of batch size).
    # Final shape: (batch_size, height, width, channels) = (N, 28, 28, 1).
    # This ensures the data is in the proper 4D format required by following CNNs (NHWC).
    x_train = np.array([_image_preprocess_single(img) for img in x_train]).reshape(-1,28,28,1)
    print(f"-- x_train preprocessing completed: {len(x_train)} images with shape: {x_train.shape}")
    x_val = np.array([_image_preprocess_single(img) for img in x_val]).reshape(-1,28,28,1)
    print(f"-- x_val preprocessing completed: {len(x_val)} images with shape: {x_val.shape}")
    x_test = np.array([_image_preprocess_single(img) for img in x_test]).reshape(-1,28,28,1)
    print(f"-- x_test preprocessing completed: {len(x_test)} images with shape: {x_test.shape}")
    print("-" * 50)

    return (x_train, y_train, x_val, y_val, x_test, y_test)

# -------------------------------
# Step 1.3: One-hot encode labels
# -------------------------------
def apply_one_hot(dataset):
    """
    Apply one-hot encoding to dataset labels
    
    Parameters:
        dataset (tuple): Dataset tuple (x_train, y_train, x_val, y_val, x_test, y_test)
        
    Returns:
        tuple: Dataset with one-hot encoded labels
    """
    (x_train, y_train, x_val, y_val, x_test, y_test) = dataset

    # Label one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    print(f"\n\nOne-hot encoding completed: ")
    print("-" * 50)
    print(f"y_train encoded shape: {y_train.shape}")
    print(f"y_val encoded shape: {y_val.shape}")
    print(f"y_test encoded shape: {y_test.shape}")
    print("-" * 50)

    return (x_train, y_train, x_val, y_val, x_test, y_test)

# -------------------------------
# Step 1.4: Data Augmentation
# -------------------------------
def augment_img(dataset):
    """
    Set up data augmentation for the training dataset
    
    Parameters:
        dataset (tuple): Dataset tuple (x_train, y_train, x_val, y_val, x_test, y_test)
        
    Returns:
        tuple: Dataset with augmentation generator
    """
    (x_train, y_train, x_val, y_val, x_test, y_test) = dataset

    # There are a large number of handwritten digit styles in the MNIST dataset.
    # Data augmentation helps improve robustness during training.

    # The parameters below are relatively mild (10% shift, 10 degree rotation),
    # which avoids distorting the digits too much.
    # If the parameters are too large (e.g., rotation_range=45),
    # digits like '6' might look like '9', which would be counterproductive.
    datagen = ImageDataGenerator(
        rotation_range=10,      # +/- 10 degree rotation
        width_shift_range=0.1,  # +/- 10% horizontal shift
        height_shift_range=0.1, # +/- 10% vertical shift
        zoom_range=0.1          # 10% zoom in/out
    )
    datagen.fit(x_train)
    print(f"\n\nImage Augmentation on x_train completed.")
    print("-" * 50)
    return (x_train, y_train, x_val, y_val, x_test, y_test, datagen)

# -------------------------------
# Step 1.5: Encapsulate data loading and preprocessing
# -------------------------------
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
                 show_demo=False,
                 preprocess=True,
                 one_hot=True,
                 augument=True):
    """
    Main function to load and prepare MNIST data
    
    Parameters:
        train_size (int): Number of training samples
        val_size (int): Number of validation samples
        test_size (int): Number of test samples
        show_demo (bool): Whether to display sample images
        preprocess (bool): Whether to apply image preprocessing
        one_hot (bool): Whether to apply one-hot encoding
        augument (bool): Whether to set up data augmentation
        
    Returns:
        tuple: Prepared dataset
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
    if augument: 
        dataset = augment_img(dataset)
    return dataset