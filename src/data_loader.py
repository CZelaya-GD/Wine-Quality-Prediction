import pandas as pd

def load_wine_data(filepath):
    """
    Loads the wine quality dataset from a CSV file.

    :param filepath:
    Returns:
        A pandas Dataframe
    """

    df = pd.read_csv(filepath)
    # Drop 'Id' column if present

    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    return df
