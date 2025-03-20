import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated dataset
np.random.seed(42)
data = pd.DataFrame({
    'Area': np.random.randint(50, 200, 500),
    'Rooms': np.random.randint(1, 10, 500),
    'Age': np.random.randint(1, 50, 500),
    'Distance': np.random.uniform(1, 20, 500),
    'Bathrooms': np.random.randint(1, 5, 500),
    'Price': np.random.randint(100000, 1000000, 500)
})

# Features and target
X = data.drop(columns=['Price'])
y = data['Price']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42))
]

# Define meta-model
meta_model = LinearRegression()

# Create stacking regressor
stacking_reg = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# 1. **Apply Random Forest Regressor**
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# 2. **Apply Gradient Boosting Regressor**
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_model.fit(X_train_scaled, y_train)
y_pred_gbr = gbr_model.predict(X_test_scaled)

# 3. **Apply XGBoost Regressor**
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# 4. **Stacking Regressor**
stacking_reg.fit(X_train_scaled, y_train)
y_pred_stacking = stacking_reg.predict(X_test_scaled)

# --- Performance Comparison ---
def print_metrics(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}")
    print(f"{model_name} - MAE: {mae:.4f}")
    print(f"{model_name} - R² Score: {r2:.4f}")
    print("="*50)

# Performance of each base model
print("Random Forest Performance:")
print_metrics("Random Forest", y_test, y_pred_rf)

print("Gradient Boosting Performance:")
print_metrics("Gradient Boosting", y_test, y_pred_gbr)

print("XGBoost Performance:")
print_metrics("XGBoost", y_test, y_pred_xgb)

# Performance of Stacking Regressor
print("Stacking Regressor Performance:")
print_metrics("Stacking Regressor", y_test, y_pred_stacking)

# --- Visualization ---
# Visual comparison of predictions: True vs Predicted for each model

plt.figure(figsize=(16, 12))

# 1. Histogram of True vs Predicted Prices (Random Forest)
plt.subplot(3, 3, 1)
sns.histplot(y_test, kde=True, color='blue', label="True Prices", bins=20)
sns.histplot(y_pred_rf, kde=True, color='orange', label="Predicted Prices", bins=20)
plt.title('Random Forest: True vs Predicted Prices')
plt.legend()

# 2. Histogram of True vs Predicted Prices (Gradient Boosting)
plt.subplot(3, 3, 2)
sns.histplot(y_test, kde=True, color='blue', label="True Prices", bins=20)
sns.histplot(y_pred_gbr, kde=True, color='orange', label="Predicted Prices", bins=20)
plt.title('Gradient Boosting: True vs Predicted Prices')
plt.legend()

# 3. Histogram of True vs Predicted Prices (XGBoost)
plt.subplot(3, 3, 3)
sns.histplot(y_test, kde=True, color='blue', label="True Prices", bins=20)
sns.histplot(y_pred_xgb, kde=True, color='orange', label="Predicted Prices", bins=20)
plt.title('XGBoost: True vs Predicted Prices')
plt.legend()

# 4. Histogram of True vs Predicted Prices (Stacking)
plt.subplot(3, 3, 4)
sns.histplot(y_test, kde=True, color='blue', label="True Prices", bins=20)
sns.histplot(y_pred_stacking, kde=True, color='orange', label="Predicted Prices", bins=20)
plt.title('Stacking Regressor: True vs Predicted Prices')
plt.legend()

# 5. Scatter plot of True vs Predicted Prices (Random Forest)
plt.subplot(3, 3, 5)
plt.scatter(y_test, y_pred_rf, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', label='Perfect Prediction')
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest: True vs Predicted Prices')
plt.legend()

# 6. Scatter plot of True vs Predicted Prices (Gradient Boosting)
plt.subplot(3, 3, 6)
plt.scatter(y_test, y_pred_gbr, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', label='Perfect Prediction')
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('Gradient Boosting: True vs Predicted Prices')
plt.legend()

# 7. Scatter plot of True vs Predicted Prices (XGBoost)
plt.subplot(3, 3, 7)
plt.scatter(y_test, y_pred_xgb, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', label='Perfect Prediction')
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('XGBoost: True vs Predicted Prices')
plt.legend()

# 8. Scatter plot of True vs Predicted Prices (Stacking)
plt.subplot(3, 3, 8)
plt.scatter(y_test, y_pred_stacking, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', label='Perfect Prediction')
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('Stacking: True vs Predicted Prices')
plt.legend()

plt.tight_layout()
plt.show()

# User input
print("Please enter the details of the house:")

area = float(input("Area (in square meters): "))
rooms = int(input("Number of Rooms: "))
age = int(input("Age of the house: "))
distance = float(input("Distance to the city center (in km): "))
bathrooms = int(input("Number of Bathrooms: "))

# Creating a DataFrame from the user input
new_house = pd.DataFrame({
    'Area': [area],
    'Rooms': [rooms],
    'Age': [age],
    'Distance': [distance],
    'Bathrooms': [bathrooms]
})

# Feature scaling (same as done with training data)
new_house_scaled = scaler.transform(new_house)

# Predict the price using the best model (Stacking Regressor)
predicted_price = stacking_reg.predict(new_house_scaled)

# Print the predicted price
print(f"\nThe predicted price of the new house is: ${predicted_price[0]:,.2f}")
