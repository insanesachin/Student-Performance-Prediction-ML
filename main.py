from src.data_generation import generate_dataset
from src.data_analysis import analyze_data
from src.data_preprocessing import preprocess_data
from src.visualization import visualize_data
from src.model_training import train_model


print("\n==========================================")
print(" STUDENT PERFORMANCE PREDICTION SYSTEM")
print("==========================================")


# STEP 1: Generate Dataset
print("\nSTEP 1: GENERATING DATASET")

df = generate_dataset(n_samples=500)


# STEP 2: Exploratory Data Analysis
print("\nSTEP 2: DATA ANALYSIS")

analyze_data(df)


# STEP 3: Data Visualization
print("\nSTEP 3: DATA VISUALIZATION")

visualize_data(df)


# STEP 4: Data Preprocessing
print("\nSTEP 4: DATA PREPROCESSING")

X, y = preprocess_data(df)


# STEP 5: Model Training
print("\nSTEP 5: MACHINE LEARNING MODEL")

model = train_model(X, y)


# STEP 6: New Student Prediction
print("\nSTEP 6: NEW STUDENT PREDICTION")

new_student = [[
    6.5,   # Study hours
    85,    # Attendance
    75,    # Previous score
    9      # Assignments completed
]]

prediction = model.predict(new_student)

print("Predicted Final Score:", round(prediction[0], 2))


print("\n==========================================")
print(" PROJECT EXECUTION COMPLETED SUCCESSFULLY")
print("==========================================")