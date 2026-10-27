
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score
)

# Step 1: Load dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# Step 2: Create target categories
median_calories = df['Calories_Burned'].median()

df['Calorie_Category'] = (
    df['Calories_Burned'] >= median_calories
).astype(int)

# 0 = Low Calories
# 1 = High Calories

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

# Step 5: Create Decision Tree using Entropy
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Step 8: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
loss = log_loss(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob[:, 1])

print("DECISION TREE USING INFORMATION GAIN")
print("------------------------------------")
print("Criterion: Entropy")
print("Root Feature:", X.columns[model.tree_.feature[0]])
print("Accuracy:", round(accuracy, 4))
print("Log Loss:", round(loss, 4))
print("Confusion Matrix:\n", matrix)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Low Calories', 'High Calories']
))

print("ROC-AUC Score:", round(auc, 4))

# Step 9: Display feature importance
print("\nFeature Importance:")

for feature, importance in zip(X.columns, model.feature_importances_):
    print(feature, ":", round(importance, 4))

# Step 10: Visualize the Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=list(X.columns),
    class_names=['Low Calories', 'High Calories'],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title('Decision Tree Using Entropy and Information Gain')
plt.tight_layout()
plt.show()

