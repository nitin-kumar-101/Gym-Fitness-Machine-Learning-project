
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    log_loss, roc_auc_score
)

# Step 1: Load dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# Step 2: Create a binary target
# High = calories burned at or above the median
median_calories = df['Calories_Burned'].median()

df['Calorie_Category'] = (
    df['Calories_Burned'] >= median_calories
).astype(int)

# 0 = Low calories, 1 = High calories

# Step 3: Select input features
X = df[
    [
        'Session_Duration (hours)',
        'Weight (kg)',
        'Age',
        'Avg_BPM',
        'Workout_Frequency (days/week)'
    ]
]

y = df['Calorie_Category']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Step 7: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)
matrix = confusion_matrix(y_test, y_pred)

print("LOGISTIC REGRESSION RESULTS")
print("---------------------------")
print("Accuracy:", round(accuracy, 4))
print("Log Loss:", round(loss, 4))
print("Confusion Matrix:\n", matrix)
print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=['Low Calories', 'High Calories']
))

# Step 8: Calculate AUC
auc = roc_auc_score(y_test, y_prob[:, 1])
print("ROC-AUC Score:", round(auc, 4))

# Step 9: Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(matrix)
plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.xticks([0, 1], ['Low', 'High'])
plt.yticks([0, 1], ['Low', 'High'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix[i, j], ha='center', va='center')

plt.colorbar()
plt.tight_layout()
plt.show()

