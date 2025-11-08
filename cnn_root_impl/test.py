#!/usr/bin/env python3
from data_preparation import load_and_prepare_data

# Test the data preparation pipeline
dataset = load_and_prepare_data(show_demo=False)
print(f"Loaded dataset with {len(dataset)} components")

# Extract components
x_train, y_train, x_val, y_val, x_test, y_test = dataset
print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")
