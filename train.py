# ⚡ Bengaluru House Price Prediction with Rich Outputs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------
# 1) Load dataset
# -------------------------------
df = pd.read_csv("Bengaluru_House_Data.csv")

# Drop useless cols
df = df.drop(["area_type", "society", "availability"], axis=1)
df = df.dropna()

# Extract BHK
df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))

# Convert sqft
def convert_sqft(x):
    try:
        if "-" in str(x):
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None
df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["total_sqft"])

# -------------------------------
# 2) Add extra features
# -------------------------------
np.random.seed(42)
df["property_age"] = np.random.randint(0, 31, size=len(df))     # synthetic
df["amenities_score"] = np.random.randint(0, 11, size=len(df))  # synthetic

# Keep only needed cols
df = df[["location", "total_sqft", "bhk", "property_age", "amenities_score", "bath", "price"]]

# Encode location
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])

print("✅ Cleaned data shape:", df.shape)

# -------------------------------
# 3) EDA (Visuals)
# -------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df["price"], bins=50, kde=True, color="skyblue")
plt.title("House Price Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x="total_sqft", y="price", data=df, alpha=0.4)
plt.title("Price vs Square Footage")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="bhk", y="price", data=df)
plt.title("Price vs BHK")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 4) Features & Target
# -------------------------------
X = df.drop("price", axis=1)
y = df["price"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 5) Models
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=5)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    print(f"\n📌 {name}")
    print("R²:", round(r2, 4))
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))

results_df = pd.DataFrame(results).T
print("\n📊 Model Comparison:")
print(results_df)

# -------------------------------
# 6) Model Comparison Chart
# -------------------------------
results_df[["R2"]].plot(kind="bar", legend=False, color="teal")
plt.title("Model Performance (R² Score)")
plt.ylabel("R²")
plt.show()

# -------------------------------
# 7) Predictions vs Actuals (Random Forest)
# -------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, predictions["Random Forest"], alpha=0.5, color="orange")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices (Random Forest)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# -------------------------------
# 8) Feature Importance (Random Forest)
# -------------------------------
rf = models["Random Forest"]
importances = rf.feature_importances_
feat_names = ["location", "total_sqft", "bhk", "property_age", "amenities_score", "bath"]

plt.figure(figsize=(6,4))
pd.Series(importances, index=feat_names).sort_values().plot(kind="barh", color="green")
plt.title("Feature Importances (Random Forest)")
plt.show()