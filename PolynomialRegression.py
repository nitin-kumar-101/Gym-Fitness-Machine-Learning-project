import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# Step 2: Select input and target
X = df[['Session_Duration (hours)']]
y = df['Calories_Burned']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train simple linear regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 5: Create polynomial features (degree 3)
poly = PolynomialFeatures(degree=3, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Step 6: Train polynomial regression
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Step 7: Make predictions
y_pred = poly_model.predict(X_test_poly)

# Step 8: Evaluate polynomial model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("POLYNOMIAL REGRESSION RESULTS")
print("-----------------------------")
print("Polynomial Degree: 3")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Step 9: Generate smooth points for plotting
X_line = pd.DataFrame({
    'Session_Duration (hours)': np.linspace(
        X['Session_Duration (hours)'].min(),
        X['Session_Duration (hours)'].max(),
        300
    )
})

# Linear predictions
y_linear = linear_model.predict(X_line)

# Polynomial predictions
X_line_poly = poly.transform(X_line)
y_polynomial = poly_model.predict(X_line_poly)

# Step 10: Plot both models
plt.figure(figsize=(10, 6))

plt.scatter(
    X_test['Session_Duration (hours)'],
    y_test,
    label='Actual Data'
)

plt.plot(
    X_line['Session_Duration (hours)'],
    y_linear,
    label='Linear Regression'
)

plt.plot(
    X_line['Session_Duration (hours)'],
    y_polynomial,
    label='Polynomial Regression (Degree 3)'
)
plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()

