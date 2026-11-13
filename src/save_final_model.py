
import joblib
import shutil

source = "models/random_forest.joblib"
destination = "models/car_price_model.joblib"

shutil.copy2(source, destination)

print("Final model saved successfully!")
print("Model:", destination)
