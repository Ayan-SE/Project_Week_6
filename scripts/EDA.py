import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def dataset_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the dataset, including the number of rows, columns, and data types.
    """
    print("\nDataset Summary:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nFirst few rows of the dataset:")
    print(df.head())

def analyze_dataset(file_path: str) -> None:
    """
    Load a dataset and display its structure.
    """
    print(f"Loading dataset from: {file_path}")
    try:
        df = load_dataset(file_path)
        dataset_summary(df)
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_central_tendency(df: pd.DataFrame) -> None:
    """
    Calculate and display the central tendency (mean, median, and mode) 
    for numeric columns in the dataset.
    """
    print("\nCentral Tendency of Numeric Columns:")
    numeric_columns = df.select_dtypes(include=['number'])   
    if numeric_columns.empty:
        print("No numeric columns found in the dataset.")
        return
    for column in numeric_columns.columns:
        mean = numeric_columns[column].mean()
        median = numeric_columns[column].median()
        mode = numeric_columns[column].mode().iloc[0] if not numeric_columns[column].mode().empty else None
        print(f"\nColumn: {column}")
        print(f"  Mean: {mean}")
        print(f"  Median: {median}")
        print(f"  Mode: {mode}")

def calculate_dispersion(df: pd.DataFrame) -> None:
    """
    Calculate and display measures of dispersion (variance, standard deviation, range, and IQR)
    for numeric columns in the dataset.
    """
    print("\nMeasures of Dispersion for Numeric Columns:")
    numeric_columns = df.select_dtypes(include=['number']) 
    if numeric_columns.empty:
        print("No numeric columns found in the dataset.")
        return
    for column in numeric_columns.columns:
        variance = numeric_columns[column].var()
        std_dev = numeric_columns[column].std()
        data_range = numeric_columns[column].max() - numeric_columns[column].min()
        q1 = numeric_columns[column].quantile(0.25)
        q3 = numeric_columns[column].quantile(0.75)
        iqr = q3 - q1
        print(f"\nColumn: {column}")
        print(f"  Variance: {variance}")
        print(f"  Standard Deviation: {std_dev}")
        print(f"  Range: {data_range}")
        print(f"  Interquartile Range (IQR): {iqr}")

def analyze_distribution_shape(df: pd.DataFrame) -> None:
    """
    Analyze and display the shape of the dataset’s distribution (skewness and kurtosis)
    for numeric columns.
    """
    print("\nShape of the Dataset's Distribution:")
    numeric_columns = df.select_dtypes(include=['number'])
    
    if numeric_columns.empty:
        print("No numeric columns found in the dataset.")
        return

    for column in numeric_columns.columns:
        column_skewness = skew(numeric_columns[column], nan_policy='omit')
        column_kurtosis = kurtosis(numeric_columns[column], nan_policy='omit')
        
        print(f"\nColumn: {column}")
        print(f"  Skewness: {column_skewness:.4f} (Positive: Right skewed, Negative: Left skewed, 0: Symmetrical)")
        print(f"  Kurtosis: {column_kurtosis:.4f} (Positive: Heavy tails, Negative: Light tails, 0: Normal)")

def visualize_distributions(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of numerical features to identify patterns, skewness, and outliers.
    Creates histograms and boxplots for each numeric column.
    """
    numeric_columns = df.select_dtypes(include=['number'])
    
    if numeric_columns.empty:
        print("No numeric columns found in the dataset.")
        return

    for column in numeric_columns.columns:
        # Set up the figure
        plt.figure(figsize=(14, 6))

        # Plot Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(numeric_columns[column], kde=True, bins=30, color='blue')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        # Plot Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=numeric_columns[column], color='orange')
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)

        # Show the plots
        plt.tight_layout()
        plt.show()

def analyze_categorical_features(df: pd.DataFrame) -> None:
    """
    Analyze and visualize the distribution of categorical features.
    Displays frequency counts and bar charts for each categorical column.
    """
    categorical_columns = df.select_dtypes(include=['object', 'category'])
    
    if categorical_columns.empty:
        print("No categorical columns found in the dataset.")
        return

    for column in categorical_columns.columns:
        # Frequency count
        print(f"\nFrequency distribution for '{column}':")
        print(df[column].value_counts())

        # Bar plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, y=column, order=df[column].value_counts().index, palette='viridis')
        plt.title(f"Bar Chart of {column}")
        plt.xlabel("Frequency")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()

def correlation_analysis(df: pd.DataFrame) -> None:
    """
    Perform and visualize correlation analysis for numerical features in the dataset.
    Displays a correlation matrix and a heatmap.
    """
    numeric_columns = df.select_dtypes(include=['number'])
    
    if numeric_columns.empty:
        print("No numeric columns found in the dataset for correlation analysis.")
        return

    # Calculate correlation matrix
    correlation_matrix = numeric_columns.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    
def check_missing_values(data):
  """
  Checks for missing values in a pandas DataFrame and provides a summary.
  """
  missing_values = pd.DataFrame(index=data.columns)
  missing_values['Missing Count'] = data.isnull().sum()
  missing_values['Missing Percentage'] = (data.isnull().sum() / len(data)) * 100

def impute_categorical_missingValue_with_mode(df, columns):
      """
      Imputes missing values in categorical columns with their respective modes.
      """
      df = df.copy()  # Avoid modifying the original DataFrame
      for col in columns:
        mode_value = df[col].mode()[0]  # Find the mode of the column
        df[col] = df[col].fillna(mode_value) 
      return df

def impute_categorical_missing_with_placeholder(df, columns,placeholder='Unknown'):
      """
       Imputes missing values in categorical columns with a specified placeholder.
      """
      df = df.copy()  # Avoid modifying the original DataFrame
      for col in columns:
        df[col] = df[col].fillna(placeholder)
      return df
