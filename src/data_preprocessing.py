
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(input_path):

    df = pd.read_csv(input_path)

    # Target variable
    X = df.drop(columns=["priceusd"])
    y = df["priceusd"]

    # Numerical features
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

    # Categorical features
    categorical_features = X.select_dtypes(include=["str", "object"]).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save prepared data and preprocessor
    joblib.dump(preprocessor, "models/preprocessor.joblib")

    print("Preprocessing completed!")
    print("Training shape:", X_train_processed.shape)
    print("Testing shape:", X_test_processed.shape)

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":

    prepare_data("data/cars_featured.csv")
