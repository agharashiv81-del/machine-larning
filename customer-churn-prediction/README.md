# Customer Churn Prediction

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to **churn (leave the company)** or **stay**.

The model uses customer information such as:

* Customer Tenure
* Monthly Charges
* Contract Type

The trained model is connected with a **Streamlit web application** where users can enter customer details and get a prediction.

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit

## 🤖 Machine Learning Model

This project uses **Logistic Regression** for customer churn classification.

The contract type is converted from text into numerical values using **LabelEncoder**.

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── train_model.py
├── model.pkl
├── contract_encoder.pkl
└── README.md
```

## ⚙️ How It Works

1. Customer data is prepared using Pandas.
2. Contract type is converted into numerical values.
3. The data is divided into training and testing sets.
4. A Logistic Regression model is trained.
5. The trained model is saved using Joblib.
6. Streamlit loads the saved model.
7. User enters customer information.
8. The application predicts whether the customer is likely to churn or stay.

## ▶️ How to Run

### 1. Install required libraries

```bash
pip install pandas scikit-learn joblib streamlit
```

### 2. Train the model

```bash
python train_model.py
```

This creates:

```text
model.pkl
contract_encoder.pkl
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Input Features

| Feature         | Description                                               |
| --------------- | --------------------------------------------------------- |
| Tenure          | Number of months the customer has stayed with the company |
| Monthly Charges | Amount paid by the customer each month                    |
| Contract Type   | Customer's contract type                                  |

## 🎯 Output

The application gives one of two predictions:

* **Customer is likely to churn**
* **Customer is likely to stay**

## 📚 Purpose

This project was created as a practical Machine Learning project to understand:

* Classification
* Logistic Regression
* Label Encoding
* Model Saving with Joblib
* Streamlit
* ML model deployment workflow

## 👨‍💻 Author

**Shiv Aghara**

BCA Student | Aspiring Data Scientist
