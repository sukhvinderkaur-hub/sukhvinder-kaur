# sukhvinder-kaur
code tech AI projects 
# =====================================================
# HOUSE PRICE PREDICTION
# CODTECH INTERNSHIP PROJECT
# =====================================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ----------------------------
# Load Dataset
# ----------------------------
try:
    data = pd.read_csv("house_prices.csv")
except FileNotFoundError:
    print("Error: house_prices.csv file not found.")
    exit()

print("\nFirst 5 Records")
print(data.head())

print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())

# ----------------------------
# Data Cleaning
# ----------------------------
data = data.dropna()

# ----------------------------
# Data Visualization
# ----------------------------

# Price Distribution
plt.figure(figsize=(6,4))
plt.hist(data["Price"], bins=20)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.show()

# Area vs Price
plt.figure(figsize=(6,4))
plt.scatter(data["Area"], data["Price"])
plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

# ----------------------------
# Feature Selection
# ----------------------------
X = data[["Area","Bedrooms","Bathrooms"]]
y = data["Price"]

# ----------------------------
# Train Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Model Training
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# Model Evaluation
# ----------------------------
print("\nModel Performance")

print("R2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))

# ----------------------------
# Save Model
# ----------------------------
joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully.")

# ----------------------------
# Predict New House Price
# ----------------------------

print("\nEnter House Details")

area = float(input("Area (sq ft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))

prediction = model.predict([[area, bedrooms, bathrooms]])

print("\nPredicted House Price = ₹{:,.2f}".format(prediction[0]))
