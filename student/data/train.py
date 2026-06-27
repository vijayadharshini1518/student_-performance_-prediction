print("Model training started")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# SAMPLE DATA
data = {
    "Math": [80, 60, 70, 90, 50, 85],
    "Science": [85, 65, 75, 95, 55, 88],
    "English": [78, 60, 72, 88, 50, 80],
    "Performance": [85, 65, 75, 95, 55, 86]
}

df = pd.DataFrame(data)

# FEATURES (INPUT)
X = df[["Math", "Science", "English"]]

# TARGET (OUTPUT)
y = df["Performance"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestRegressor()

# TRAIN MODEL
model.fit(X_train, y_train)

# accuracy
accuracy = model.score(X_test, y_test)

print("Model trained successfully")
print("Accuracy:", round(accuracy, 2))

# save model
joblib.dump(model, "model.pkl")
print("Model saved")