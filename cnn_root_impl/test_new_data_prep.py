#!/usr/bin/env python3
"""
Test script for the new data preparation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_prep.data_preparation import load_dataset, DATASET_LEVELS


def test_level_loading():
    """Test loading different dataset levels"""
    print("Testing dataset level loading...")
    
    # Test loading level_1 dataset
    print("\n1. Testing level_1 dataset:")
    data = load_dataset(level='level_1')
    print(f"   Loaded data shapes: {[d.shape for d in data]}")
    
    # Test loading with custom size
    print("\n2. Testing custom size dataset (500 training, 100 testing samples):")
    data = load_dataset(train_size=500, test_size=100)
    print(f"   Loaded data shapes: {[d.shape for d in data]}")
    
    # Test loading with augmentation
    print("\n3. Testing level_2 dataset with augmentation:")
    data = load_dataset(level='level_2', augment=True)
    print(f"   Loaded data shapes: {[d.shape for d in data]}")


if __name__ == "__main__":
    test_level_loading()
    print("\nTest completed successfully!")