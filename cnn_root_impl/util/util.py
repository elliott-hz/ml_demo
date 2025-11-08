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