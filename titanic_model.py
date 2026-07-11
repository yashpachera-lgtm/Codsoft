import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("titanic.csv")
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

sns.barplot(x="Sex", y="Survived", data=data)
plt.title("Survival Rate by Sex")
plt.show()

sns.barplot(x="Pclass", y="Survived", data=data)
plt.title("Survival Rate by Passenger Class")
plt.show()

data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)
X = data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]]
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

importance = model.coef_[0]
features = X.columns
plt.figure(figsize=(8,5))
plt.bar(features, importance, color="green")
plt.xticks(rotation=45)
plt.title("Feature Importance in Survival Prediction")
plt.ylabel("Coefficient Value")
plt.show()

sample_passenger = pd.DataFrame(columns=X.columns)
sample_passenger.loc[0] = [3, 22, 1, 0, 7.25, 1, 0, 1]
print("Survival Prediction:", model.predict(sample_passenger))

test_data = pd.read_csv("test.csv")
test_data["Age"] = test_data["Age"].fillna(data["Age"].median())
test_data["Fare"] = test_data["Fare"].fillna(data["Fare"].median())
test_data = pd.get_dummies(test_data, columns=["Sex"], drop_first=True)

for col in ["Sex_male", "Embarked_Q", "Embarked_S"]:
    if col not in test_data.columns:
        test_data[col] = 0

X_test_final = test_data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]]
test_predictions = model.predict(X_test_final)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})
submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully!")
