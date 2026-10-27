
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

print(df.head())
print(df.info())

# Step 2: Create target categories
median_calories = df['Calories_Burned'].median()

df['Calorie_Category'] = np.where(
    df['Calories_Burned'] >= median_calories,
    'High Calories',
    'Low Calories'
)

# Step 3: Select input features
X = df[
    [
        'Session_Duration (hours)',
        'Avg_BPM',
        'Weight (kg)',
        'Age'
    ]
]

y = df['Calorie_Category']

# Step 4: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=55, stratify=y
)

# Step 5: Normalize the features
scaler = MinMaxScaler()

X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print("\nOriginal Data Summary:")
print(X.describe())

print("\nAfter Normalization:")
print(pd.DataFrame(
    X_train_norm,
    columns=X.columns
).describe())

# Step 6: Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_norm, y_train)

# Step 7: Predict test data
pr = knn.predict(X_test_norm)

# Step 8: Calculate confusion matrix
tab = confusion_matrix(
    y_test, pr,
    labels=['Low Calories', 'High Calories']
)

print("\nConfusion Matrix:\n", tab)

# Step 9: Calculate accuracy
acc = accuracy_score(y_test, pr) * 100

print("\nAccuracy: {:.2f}%".format(acc))

# Step 10: Plot confusion matrix
sns.heatmap(
    tab,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Low Calories', 'High Calories'],
    yticklabels=['Low Calories', 'High Calories']
)

plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('KNN Confusion Matrix')
plt.show()

