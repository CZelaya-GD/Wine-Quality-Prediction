from src.data_loader import load_wine_data
from src.preprocess import preprocess_data, create_new_features
from src.model import WineQualityModel
from src.eda import analyze_data


def run_training(data_path, run_eda=False):
    """
    Main training pipeline with EDA.
    """

    df = load_wine_data(data_path)

    if run_eda:

        analyze_data(df)  # Run EDA if specified

    df = create_new_features(df)  # Feature engineering

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = WineQualityModel(model_type='random_forest', max_depth=10)
    model.train(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    print(f"Test Accuracy: {accuracy:.2f}")
    return model, scaler
