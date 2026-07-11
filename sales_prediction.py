import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("advertising.csv")
X = data[["TV"]]   # predictor variable
y = data["Sales"]  # target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
indices = np.arange(len(y_test))
plt.bar(indices, y_test, width=0.4, label="Actual Sales", color="blue")
plt.bar(indices + 0.4, y_pred, width=0.4, label="Predicted Sales", color="red")
plt.xlabel("Test Samples")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales (Simple Linear Regression)")
plt.legend()
plt.show()
new_data = pd.DataFrame({"TV":[100]})
print("Predicted Sales:", model.predict(new_data)[0])
