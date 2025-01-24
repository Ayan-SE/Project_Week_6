import pandas as pd

def sum_transactions_by_customer(df: pd.DataFrame, customer_col: str, amount_col: str) -> pd.DataFrame:
    """
    Calculate the sum of all transaction amounts for each customer.
    """
    if customer_col not in df.columns or amount_col not in df.columns:
        raise ValueError("Specified columns not found in the DataFrame.")

    # Group by customer and sum the transaction amounts
    result = df.groupby(customer_col)[amount_col].sum().reset_index()

    # Rename columns for clarity
    result.columns = ['Customer', 'Total Transaction Amount']

    return result

def average_transaction_by_customer(df: pd.DataFrame, customer_col: str, amount_col: str) -> pd.DataFrame:
    """
    Calculate the average transaction amount for each customer.
    """
    if customer_col not in df.columns or amount_col not in df.columns:
        raise ValueError("Specified columns not found in the DataFrame.")
    
    # Group by customer and calculate the mean transaction amount
    result = df.groupby(customer_col)[amount_col].mean().reset_index()

    # Rename columns for clarity
    result.columns = ['Customer', 'Average Transaction Amount']

    return result

def average_transaction_by_customer(df: pd.DataFrame, customer_col: str, amount_col: str) -> pd.DataFrame:
    """
    Calculate the average transaction amount for each customer.
    """
    if customer_col not in df.columns or amount_col not in df.columns:
        raise ValueError("Specified columns not found in the DataFrame.")
    
    # Group by customer and calculate the mean transaction amount
    result = df.groupby(customer_col)[amount_col].mean().reset_index()

    # Rename columns for clarity
    result.columns = ['Customer', 'Average Transaction Amount']

    return result

def transaction_std_by_customer(df: pd.DataFrame, customer_col: str, amount_col: str) -> pd.DataFrame:
    """
    Calculate the standard deviation of transaction amounts for each customer.
    """
    if customer_col not in df.columns or amount_col not in df.columns:
        raise ValueError("Specified columns not found in the DataFrame.")

    # Group by customer and calculate the standard deviation of transaction amounts
    result = df.groupby(customer_col)[amount_col].std().reset_index()

    # Rename columns for clarity
    result.columns = ['Customer', 'Transaction Amount Standard Deviation']

    return result
