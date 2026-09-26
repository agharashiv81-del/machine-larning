import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


data = {
    "tenure": [1, 2, 5, 10, 15, 20, 3, 7, 12, 25],
    "monthly_charges": [80, 90, 70, 50, 45, 40, 85, 65, 55, 35],
    "contract": [
        "Month-to-month", "Month-to-month", "Month-to-month",
        "One year", "One year", "Two year",
        "Month-to-month", "One year", "One year", "Two year"
    ],
    "churn": [
        "Yes", "Yes", "Yes", "No", "No",
        "No", "Yes", "No", "No", "No"
    ]
}


df = pd.DataFrame(data)


contract_encoder = LabelEncoder()

df["contract"] = contract_encoder.fit_transform(df["contract"])

df["churn"] = df["churn"].map({
    "No": 0,
    "Yes": 1
})


x = df[["tenure", "monthly_charges", "contract"]]

y = df["churn"]


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression()

model.fit(x_train, y_train)


y_pred = model.predict(x_test)


print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


joblib.dump(model, "model.pkl")

joblib.dump(contract_encoder, "contract_encoder.pkl")


print("Model Saved Successfully.....")