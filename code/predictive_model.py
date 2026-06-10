import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("../dataset/titanic.csv")

data["age"] = data["age"].fillna(data["age"].median())
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])

data = pd.get_dummies(data, columns=["sex", "embarked"], drop_first=True)

x = data[["pclass", "age", "fare", "sex_male"]]
y = data["survived"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

pred = model.predict(x_test)

score = accuracy_score(y_test, pred)

print("Accuracy:", score)
