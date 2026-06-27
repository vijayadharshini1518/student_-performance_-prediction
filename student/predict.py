import joblib
import numpy as np

# load trained model
model = joblib.load("model.pkl")

print("Enter student data:")

hours = float(input("Hours studied: "))

# prediction
result = model.predict([[hours]])

print("Predicted Marks:", round(result[0], 2))