import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histograms(data, num_cols, bins=10, figsize=(4, 2)):
     """
     Plots histograms for specified numerical columns in a DataFrame.
    """
     for col in num_cols:
       plt.figure(figsize=figsize)
       plt.hist(data[col], bins=bins)
       plt.xlabel(col)
       plt.ylabel('Frequency')
       plt.title(f'Histogram of {col}')
       plt.show()

def plot_bar_charts(df, cat_cols, figsize=(10, 5)):
  """
  Plots bar charts for specified categorical columns in a DataFrame.
  """
  for col in cat_cols:
    # Check if the column is categorical
    if df[col].dtype == 'object' or df[col].dtype == 'category': 
      plt.figure(figsize=figsize)
      df[col].value_counts().plot(kind='bar')
      plt.xlabel(col)
      plt.ylabel('Count')
      plt.title(f'Bar Chart of {col}')
      plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
      plt.show()
    else:
      print(f"Column '{col}' is not categorical. Skipping bar chart.")

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