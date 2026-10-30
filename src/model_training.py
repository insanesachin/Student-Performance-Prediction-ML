import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt
import os


def train_model(X, y):

    print("\n========== MODEL TRAINING ==========")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Initialize model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")

    print("MAE:", round(mae, 3))
    print("MSE:", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

    # Actual vs Predicted
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.scatter(y_test, y_pred)

    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")

    plt.title("Actual vs Predicted Scores")

    plt.savefig("outputs/model_performance.png")
    plt.close()

    # Feature coefficients
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    print("\nFeature Coefficients:")
    print(coefficients)

    return model