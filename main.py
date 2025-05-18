from src.train import run_training

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/WineQT.csv"
    run_training(data_path, run_eda=True)
