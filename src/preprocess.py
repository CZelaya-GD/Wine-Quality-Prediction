from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Splits data into features (X) and label (y), then splits into train and test sets.
    Standardizes features for better model performance.
    """

    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def create_new_features(df):
    """
    Example of feature engineering.
    """

    df["alcohol_sulphates"] = df["alcohol"] * df["sulphates"]

    return df
