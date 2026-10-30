import numpy as np
import pandas as pd
import os


def generate_dataset(n_samples=500):

    np.random.seed(42)

    # Generate features
    study_hours = np.random.uniform(1, 10, n_samples)

    attendance = np.random.uniform(40, 100, n_samples)

    previous_score = np.random.uniform(30, 100, n_samples)

    assignments_completed = np.random.randint(1, 11, n_samples)

    # Generate random noise
    noise = np.random.normal(0, 5, n_samples)

    # Calculate final score
    final_score = (
        2.5 * study_hours
        + 0.20 * attendance
        + 0.35 * previous_score
        + 1.2 * assignments_completed
        + noise
    )

    final_score = np.clip(final_score, 0, 100)

    # Create DataFrame
    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments_completed,
        "final_score": final_score
    })

    # Introduce missing values
    for column in ["study_hours", "attendance", "previous_score"]:

        missing_indices = np.random.choice(
            df.index,
            size=5,
            replace=False
        )

        df.loc[missing_indices, column] = np.nan

    # Save dataset
    os.makedirs("datasets", exist_ok=True)

    df.to_csv(
        "datasets/student_data.csv",
        index=False
    )

    print("Dataset generated successfully!")
    print("Total records:", len(df))

    return df


if __name__ == "__main__":

    generate_dataset()