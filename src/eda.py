import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_data(df, output_path="eda_plots"):
    """
    Performs EDA and saves plots.
    """

    # Ensure directory exists
    import os
    os.makedirs(output_path, exist_ok=True)

    # Quality distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='quality', data=df)
    plt.title('Distribution of Wine Quality')
    plt.savefig(f"{output_path}/quality_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(f"{output_path}/correlation_heatmap.png")
    plt.close()

    # Distribution of features
    df.hist(bins=20, figsize=(12, 10))
    plt.suptitle('Histograms of Features', y=0.92)
    plt.savefig(f"{output_path}/feature_histograms.png")
    plt.close()

    print(f"EDA plots saved to '{output_path}'")
