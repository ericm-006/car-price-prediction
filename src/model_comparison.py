
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/model_results.csv")

print("MODEL RANKING:")
print(df[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Model", y="MAE")

plt.title("Model Comparison - MAE")
plt.ylabel("Mean Absolute Error (USD)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("data/model_comparison.png")
plt.show()

print("\nBest model:", df.loc[df["MAE"].idxmin(), "Model"])
