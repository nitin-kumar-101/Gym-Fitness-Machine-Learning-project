import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# Step 2: Select multiple input features
X = df[
    [
        'Session_Duration (hours)',
        'Weight (kg)',
        'Age',
        'Avg_BPM',
        'Workout_Frequency (days/week)'
    ]
]

# Step 3: Select the target variable
y = df['Calories_Burned']

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MULTIPLE LINEAR REGRESSION RESULTS")
print("----------------------------------")
print("Input Features:", list(X.columns))
print("Target: Calories_Burned")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R-squared (R2 Score):", round(r2, 4))


# Step 9: Plot actual vs predicted calories
plt.figure(figsize=(12,8))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Calories Burned')
plt.ylabel('Predicted Calories Burned')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.show()

