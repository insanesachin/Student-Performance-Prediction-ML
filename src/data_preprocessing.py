import pandas as pd

from sklearn.impute import SimpleImputer


def preprocess_data(df):

    print("\n========== DATA PREPROCESSING ==========")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Separate features and target
    X = df.drop(columns=["final_score"])
    y = df["final_score"]

    # Handle missing feature values
    imputer = SimpleImputer(strategy="median")

    X_imputed = imputer.fit_transform(X)

    X = pd.DataFrame(
        X_imputed,
        columns=X.columns,
        index=X.index
    )

    # Handle missing target values, if any
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print("Missing values handled.")
    print("Duplicates removed.")
    print("Final dataset shape:", X.shape)

    return X, y