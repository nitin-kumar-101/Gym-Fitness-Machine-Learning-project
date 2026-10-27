
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import DecisionBoundaryDisplay

# Load dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# Create target categories
median_calories = df['Calories_Burned'].median()

df['Calorie_Category'] = (
    df['Calories_Burned'] >= median_calories
).astype(int)

# 0 = Low Calories
# 1 = High Calories

# Select two features for visualization
X = df[['Session_Duration (hours)', 'Avg_BPM']].values
y = df['Calorie_Category'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict test data
y_pred = svm.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Display metrics
print("SVM CLASSIFIER RESULTS")
print("----------------------")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# Plot decision boundary
DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method='predict',
    alpha=0.8,
    cmap='Pastel1',
    xlabel='Session Duration (hours)',
    ylabel='Average BPM'
)

# Plot actual data points
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color='purple',
    s=30,
    edgecolors='black',
    label='Low Calories (0)'
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color='yellow',
    s=30,
    edgecolors='black',
    label='High Calories (1)'
)

plt.title('SVM Decision Boundary - Gym Dataset')
plt.legend()
plt.show()

