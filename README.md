
# Used Car Price Prediction

## Project Description

This project uses Machine Learning regression algorithms to predict the prices of used cars based on their characteristics.

## Dataset

The dataset contains 56,244 used car records with information about:

- Make and model
- Production year
- Mileage
- Fuel type
- Engine volume
- Condition
- Color
- Transmission
- Drive unit
- Segment

Target variable: `priceUSD`

## Project Structure

car-price-prediction/
- data/
- notebooks/
- src/
  - data_cleaning.py
  - feature_engineering.py
  - data_preprocessing.py
  - model_training.py
  - model_evaluation.py
  - model_comparison.py
  - save_final_model.py
  - predict_price.py
- models/
  - car_price_model.joblib
- README.md
- requirements.txt

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd car-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run scripts in this order:

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/data_preprocessing.py
python src/model_training.py
python src/model_evaluation.py
python src/model_comparison.py
python src/save_final_model.py
python src/predict_price.py
```

## Models Tested

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Model Evaluation

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Random Forest | 1093.42 | 2611.77 | 0.8988 |
| Decision Tree | 1301.77 | 3129.92 | 0.8547 |
| Gradient Boosting | 1529.30 | 2918.05 | 0.8737 |
| Linear Regression | 1972.61 | 3905.53 | 0.7738 |

## Final Model

Random Forest Regressor was selected as the final model because it achieved the lowest MAE and highest R2.

The model has an average absolute prediction error of approximately $1,093.

## Example Prediction

Volkswagen Golf, 2014:

Predicted price: $12,534.12
