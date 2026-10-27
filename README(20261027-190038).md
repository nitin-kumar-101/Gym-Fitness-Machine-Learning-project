# Gym & Fitness Data Analysis and Machine Learning

## Project Overview
This project explores a gym and fitness dataset using exploratory data analysis (EDA), data visualization, and beginner-friendly machine learning models. It examines workout patterns and predicts or groups records using fitness-related attributes.

The dataset includes age, gender, weight, height, BMI, heart rate, workout duration, calories burned, workout type, fat percentage, water intake, workout frequency, and experience level.

## Objectives
- Explore and understand the dataset.
- Visualize relationships between workout features and calories burned.
- Build regression models to estimate calories burned.
- Classify records as **High Calories** or **Low Calories**.
- Group records using K-Means clustering.
- Compare models with appropriate evaluation metrics.

## Machine Learning Models
1. **Simple Linear Regression** — predicts calories burned from session duration.
2. **Multiple Linear Regression** — predicts calories burned using multiple numerical features.
3. **Polynomial Regression** — models a polynomial relationship between session duration and calories burned.
4. **Logistic Regression** — classifies records as High Calories or Low Calories.
5. **Decision Tree Classifier** — uses entropy-based splitting for classification.
6. **Support Vector Machine (SVM)** — classifies records using selected workout features.
7. **K-Means Clustering** — groups records into three clusters based on session duration and calories burned.
8. **K-Nearest Neighbors (KNN)** — classifies records using selected features and normalized data.

## Dataset
Expected cleaned dataset filename: `gym_dataset_cleaned.csv`

Expected columns:
- `Age`
- `Gender`
- `Weight (kg)`
- `Height (m)`
- `BMI`
- `Max_BPM`
- `Avg_BPM`
- `Resting_BPM`
- `Session_Duration (hours)`
- `Calories_Burned`
- `Workout_Type`
- `Fat_Percentage`
- `Water_Intake (liters)`
- `Workout_Frequency (days/week)`
- `Experience_Level`

Put the CSV file in the same folder as the Python scripts, or update the file path in the code.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Spyder IDE

## How to Run
1. Install Python and the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Clone or download the repository.
3. Put `gym_dataset_cleaned.csv` in the project folder.
4. Open a script in Spyder or another Python IDE and run it.

To run the model comparison script from a terminal:
```bash
python model_comparison.py
```

The comparison script saves:
- `regression_model_comparison.csv`
- `classification_model_comparison.csv`

It also displays model-comparison charts and a K-Means clustering plot.

## Evaluation Metrics

### Regression
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination

Lower MAE, MSE, and RMSE generally indicate smaller prediction errors. A higher R² generally indicates that the model explains more variation in the target variable.

### Classification
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

The classification target is created using the median of `Calories_Burned`: records at or above the median are labeled **High Calories**, while records below it are labeled **Low Calories**. These are relative dataset categories, not medical thresholds.

### Clustering
K-Means is assessed by examining cluster sizes and cluster centers. Classification metrics are not directly applicable to unsupervised clustering without reference labels.

## Suggested Project Structure
```text
gym-fitness-ml-project/
├── gym_dataset_cleaned.csv
├── model1_simple_linear_regression.py
├── model2_multiple_linear_regression.py
├── model3_polynomial_regression.py
├── model4_logistic_regression.py
├── model5_decision_tree.py
├── model6_svm.py
├── model7_kmeans.py
├── model8_knn.py
├── model_comparison.py
└── README.md
```

The filenames above are suggestions. Use the actual filenames of the scripts you upload.

## Notes and Limitations
- Results depend on the dataset, features, and train/test split.
- High Calories and Low Calories are defined relative to the dataset median.
- The models may use different features, so comparisons should be interpreted carefully.
- K-Means clusters do not automatically represent fitness levels or medical categories.
- This project is for educational purposes and is not intended for medical advice.

## Author
Created as a student machine learning project.

## License
No license has been specified. Add an open-source license if you want to define how others may use, modify, or distribute this project.
