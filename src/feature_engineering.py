
import pandas as pd
from datetime import datetime


def engineer_features(input_path, output_path):

    df = pd.read_csv(input_path)

    current_year = datetime.now().year

    # Car age
    df["car_age"] = current_year - df["year"]

    # Mileage per year
    df["mileage_per_year"] = (
        df["mileage_kilometers"] / df["car_age"].clip(lower=1)
    )

    # Engine volume in liters
    df["engine_volume_liters"] = df["volume_cm3"] / 1000

    # Newer car indicator
    df["is_newer_car"] = (df["year"] >= 2015).astype(int)

    # High mileage indicator
    df["is_high_mileage"] = (
        df["mileage_kilometers"] >= 200000
    ).astype(int)

    # Brand and model combination
    df["brand_model"] = (
        df["make"].astype(str) + "_" + df["model"].astype(str)
    )

    df.to_csv(output_path, index=False)

    print("Feature engineering completed!")
    print("Final shape:", df.shape)
    print("\nNew columns:")
    print([
        "car_age",
        "mileage_per_year",
        "engine_volume_liters",
        "is_newer_car",
        "is_high_mileage",
        "brand_model"
    ])


if __name__ == "__main__":
    engineer_features(
        "data/cars_cleaned.csv",
        "data/cars_featured.csv"
    )
