
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("data/cars_featured.csv")

X = df.drop(columns=["priceusd"])
y = df["priceusd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_files = {
    "Linear Regression": "models/linear_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "Random Forest": "models/random_forest.joblib",
    "Gradient Boosting": "models/gradient_boosting.joblib"
}

results = []

for name, path in model_files.items():

    model = joblib.load(path)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="MAE")

print("\nMODEL EVALUATION RESULTS:")
print(results_df.to_string(index=False))

results_df.to_csv("data/model_results.csv", index=False)

print("\nResults saved to data/model_results.csv")




# Actual vs Predicted Prices

best_model = joblib.load("models/car_price_model.joblib")

predictions = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions.round(2),
})

comparison["Error"] = (
    comparison["Actual Price"] - comparison["Predicted Price"]
).abs().round(2)

print("\nACTUAL VS PREDICTED PRICES:")
print(comparison.head(10).to_string(index=False))

comparison.head(100).to_csv(
    "data/prediction_examples.csv",
    index=False
)

print("\nPrediction examples saved!")
