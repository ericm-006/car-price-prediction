
import joblib
import pandas as pd

from datetime import datetime

model = joblib.load("models/car_price_model.joblib")


def predict_car_price(
    make,
    model_name,
    year,
    condition,
    mileage,
    fuel_type,
    volume,
    color,
    transmission,
    drive_unit,
    segment
):

    current_year = datetime.now().year

    car_age = current_year - year

    data = pd.DataFrame([{
        "make": make,
        "model": model_name,
        "year": year,
        "condition": condition,
        "mileage_kilometers": mileage,
        "fuel_type": fuel_type,
        "volume_cm3": volume,
        "color": color,
        "transmission": transmission,
        "drive_unit": drive_unit,
        "segment": segment,
        "car_age": car_age,
        "mileage_per_year": mileage / max(car_age, 1),
        "engine_volume_liters": volume / 1000,
        "is_newer_car": int(year >= 2015),
        "is_high_mileage": int(mileage >= 200000),
        "brand_model": f"{make}_{model_name}"
    }])

    prediction = model.predict(data)[0]

    print(f"\nEstimated car price: ${prediction:,.2f}")

    return prediction


if __name__ == "__main__":

    predict_car_price(
        make="volkswagen",
        model_name="golf",
        year=2014,
        condition="with mileage",
        mileage=180000,
        fuel_type="diesel",
        volume=1600,
        color="black",
        transmission="mechanics",
        drive_unit="front-wheel drive",
        segment="c"
    )
