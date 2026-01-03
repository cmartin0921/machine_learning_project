from ml_project.data import clean_data
from ml_project.feature_engineering import generate_features


def main():
    cleaned = clean_data(output_path="data/interim/cleaned_dataset.csv")
    print(f"Cleaned dataset shape: {cleaned.shape}")

    engineered = generate_features(
        cleaned,
        imputation_strategy="knn",
        knn_neighbors=5,
        scaling_method="standard",
        save_path="data/processed/feature_engineered_dataset.csv",
    )
    print(f"Feature-engineered dataset shape: {engineered.shape}")


if __name__ == "__main__":
    main()
