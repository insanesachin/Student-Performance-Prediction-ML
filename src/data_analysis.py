import pandas as pd


def analyze_data(df):

    print("\n========== DATASET OVERVIEW ==========")

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Information:")
    df.info()

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nCorrelation Matrix:")
    print(df.corr(numeric_only=True))

    return df