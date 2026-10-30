import matplotlib.pyplot as plt
import seaborn as sns
import os


def visualize_data(df):

    os.makedirs("outputs", exist_ok=True)

    # 1. Study Hours vs Final Score
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="study_hours",
        y="final_score"
    )

    plt.title("Study Hours vs Final Score")

    plt.savefig("outputs/study_hours.png")
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation")

    plt.savefig("outputs/correlation.png")
    plt.close()

    # 3. Final Score Distribution
    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["final_score"],
        kde=True
    )

    plt.title("Final Score Distribution")

    plt.savefig("outputs/score_distribution.png")
    plt.close()

    # 4. Attendance vs Final Score
    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=df,
        x="attendance",
        y="final_score"
    )

    plt.title("Attendance vs Final Score")

    plt.savefig("outputs/attendance_vs_score.png")
    plt.close()

    print("\nVisualizations saved in outputs folder.")