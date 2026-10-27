
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# Step 2: Select one input feature (X) and target (y)
X = df[['Session_Duration (hours)']]
y = df['Calories_Burned']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict calories burned
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("SIMPLE LINEAR REGRESSION RESULTS")
print("--------------------------------")
print("Input Feature: Session_Duration (hours)")
print("Target: Calories_Burned")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R-squared (R2 Score):", round(r2, 4))


# Step 8: Visualize actual values and regression line
X_line = pd.DataFrame({
    'Session_Duration (hours)': np.sort(
        X_test['Session_Duration (hours)'].values
    )
})

y_line = model.predict(X_line)

plt.figure(figsize=(12,8))
plt.scatter(
    X_test['Session_Duration (hours)'],
    y_test,
    label='Actual Calories'
)

plt.plot(
    X_line['Session_Duration (hours)'],
    y_line,
    label='Regression Line'
)
plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

