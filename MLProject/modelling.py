import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    X = pd.read_csv(os.path.join(BASE_DIR, "telco_churn_preprocessing", "X.csv"))
    y = pd.read_csv(
        os.path.join(BASE_DIR, "telco_churn_preprocessing", "y.csv")
    )["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print("Accuracy:", acc)

if __name__ == "__main__":
    main()
