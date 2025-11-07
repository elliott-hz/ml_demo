#!/usr/bin/env python3
from data_preparation import load_and_prepare_data

# Test the data preparation pipeline
dataset = load_and_prepare_data(show_demo=False)
print(f"Loaded dataset with {len(dataset)} components")
