def create_incremental_combinations(columns, base_column='Close'):
    """
    Create incremental combinations: Close -> Close,High -> Close,High,Low -> etc.
    
    Args:
        columns (list): List of all column names
        base_column (str): Column that must always be included (default: 'Close')
    
    Returns:
        list: List of column combinations, each combination is a list
    """
    if base_column not in columns:
        raise ValueError(f"Base column '{base_column}' not found in columns")
    
    # Reorder so base_column comes first
    reordered = [base_column] + [col for col in columns if col != base_column]
    
    # Create incremental combinations
    combinations_list = []
    for i in range(1, len(reordered) + 1):
        combination = reordered[:i]
        combinations_list.append(combination)
    
    return combinations_list