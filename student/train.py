import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# load dataset
data = pd.read_csv("data/student.csv")

# features and target
X = data[['hours']]
y = data['marks']

# split data (better practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
model = LinearRegression()

# train model
model.fit(X_train, y_train)

# accuracy
accuracy = model.score(X_test, y_test)

print("Model trained successfully")
print("Accuracy:", round(accuracy, 2))

# save model (ONLY ONE FILE)
joblib.dump(model, "model.pkl")

print("Model saved")