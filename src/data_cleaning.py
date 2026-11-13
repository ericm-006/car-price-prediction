
import pandas as pd
import numpy as np


def clean_data(input_path, output_path):

    df = pd.read_csv(input_path)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Standardize categorical values
    categorical_columns = [
        "make", "model", "condition", "fuel_type",
        "color", "transmission", "drive_unit", "segment"
    ]

    for column in categorical_columns:
        df[column] = df[column].str.strip().str.lower()

    # Convert numerical columns
    numerical_columns = [
        "priceusd", "year", "mileage_kilometers", "volume_cm3"
    ]

    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove invalid prices
    df = df[df["priceusd"] > 0]

    # Remove impossible production years
    df = df[(df["year"] >= 1950) & (df["year"] <= 2026)]

    # Remove impossible mileage
    df = df[df["mileage_kilometers"] >= 0]

    # Remove impossible engine volumes
    df.loc[
        (df["volume_cm3"] < 500) |
        (df["volume_cm3"] > 8000),
        "volume_cm3"
    ] = np.nan

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("Cleaning completed!")
    print("Final shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    clean_data(
        "data/cars.csv",
        "data/cars_cleaned.csv"
    )
