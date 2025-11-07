#!/usr/bin/env python3
"""
Test script for data_preparation module
"""
import sys
import os

# Add the cnn_root_impl directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    # Import the load_and_prepare_data function
    from data_preparation import load_and_prepare_data
    
    print("Testing load_and_prepare_data function...")
    
    # Call the function with a small dataset for testing
    dataset = load_and_prepare_data(
        train_size=100, 
        val_size=20, 
        test_size=20,
        show_demo=False,
        preprocess=True,
        one_hot=True,
        augument=True
    )
    
    # Print information about the loaded dataset
    print("Dataset loaded successfully!")
    print(f"Number of elements in dataset tuple: {len(dataset)}")
    
    if len(dataset) >= 6:
        x_train, y_train, x_val, y_val, x_test, y_test = dataset[:6]
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"x_val shape: {x_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
    if len(dataset) == 7:
        # If data augmentation is included
        datagen = dataset[6]
        print(f"Data augmentation generator: {type(datagen)}")
        
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error occurred during testing: {e}")
    import traceback
    traceback.print_exc()