
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# ==================================================
# PART 1: REGRESSION MODEL COMPARISON
# ==================================================

X_simple = df[['Session_Duration (hours)']]
X_multiple = df[
    ['Session_Duration (hours)', 'Weight (kg)', 'Age',
     'Avg_BPM', 'Workout_Frequency (days/week)']
]
y_reg = df['Calories_Burned']

X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y_reg, test_size=0.2, random_state=42
)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multiple, y_reg, test_size=0.2, random_state=42
)

# Simple Linear Regression
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train)

pred_simple = simple_model.predict(X_test_s)

# Multiple Linear Regression
multiple_model = LinearRegression()
multiple_model.fit(X_train_m, y_train_m)

pred_multiple = multiple_model.predict(X_test_m)

# Polynomial Regression (degree 3)
poly = PolynomialFeatures(degree=3, include_bias=False)

X_train_poly = poly.fit_transform(X_train_s)
X_test_poly = poly.transform(X_test_s)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

pred_poly = poly_model.predict(X_test_poly)

# Store regression results
regression_results = []

models_reg = {
    'Simple Linear Regression': pred_simple,
    'Multiple Linear Regression': pred_multiple,
    'Polynomial Regression': pred_poly
}

for name, predictions in models_reg.items():

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    regression_results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    })

regression_df = pd.DataFrame(regression_results)

print("\n========== REGRESSION MODEL COMPARISON ==========")
print(regression_df.round(4).to_string(index=False))

regression_df.to_csv('regression_model_comparison.csv', index=False)

# Plot regression comparison
regression_df.set_index('Model')[['MAE', 'RMSE']].plot(
    kind='bar', figsize=(10, 5)
)

plt.title('Regression Models: MAE and RMSE Comparison')
plt.ylabel('Error')
plt.xlabel('Model')
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.show()

regression_df.set_index('Model')['R2 Score'].plot(
    kind='bar', figsize=(9, 5)
)

plt.title('Regression Models: R2 Score Comparison')
plt.ylabel('R2 Score')
plt.xlabel('Model')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ==================================================
# PART 2: CLASSIFICATION MODEL COMPARISON
# ==================================================

median_calories = df['Calories_Burned'].median()

df['Calorie_Category'] = np.where(
    df['Calories_Burned'] >= median_calories,
    1,
    0
)

X_class = df[
    ['Session_Duration (hours)', 'Weight (kg)', 'Age',
     'Avg_BPM', 'Workout_Frequency (days/week)']
]

y_class = df['Calorie_Category']

X_train, X_test, y_train, y_test = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

classification_results = []

def evaluate_model(name, model, X_train_data, X_test_data):
    model.fit(X_train_data, y_train)
    predictions = model.predict(X_test_data)

    classification_results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, zero_division=0),
        'Recall': recall_score(y_test, predictions, zero_division=0),
        'F1 Score': f1_score(y_test, predictions, zero_division=0)
    })


# Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
evaluate_model(
    'Logistic Regression', log_model,
    X_train_scaled, X_test_scaled
)

# Decision Tree using Entropy
tree_model = DecisionTreeClassifier(
    criterion='entropy', max_depth=5, random_state=42
)
evaluate_model(
    'Decision Tree', tree_model,
    X_train, X_test
)

# SVM using two features, matching your earlier SVM script
X_svm = df[['Session_Duration (hours)', 'Avg_BPM']]

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_svm,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

svm_scaler = StandardScaler()
X_train_svm_scaled = svm_scaler.fit_transform(X_train_svm)
X_test_svm_scaled = svm_scaler.transform(X_test_svm)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_svm_scaled, y_train_svm)

svm_predictions = svm_model.predict(X_test_svm_scaled)

classification_results.append({
    'Model': 'SVM',
    'Accuracy': accuracy_score(y_test_svm, svm_predictions),
    'Precision': precision_score(y_test_svm, svm_predictions, zero_division=0),
    'Recall': recall_score(y_test_svm, svm_predictions, zero_division=0),
    'F1 Score': f1_score(y_test_svm, svm_predictions, zero_division=0)
})

# KNN using four features, with MinMax normalization
knn_scaler = MinMaxScaler()
X_train_knn = knn_scaler.fit_transform(X_train)
X_test_knn = knn_scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=10)
evaluate_model(
    'KNN', knn_model,
    X_train_knn, X_test_knn
)

classification_df = pd.DataFrame(classification_results)

print("\n========== CLASSIFICATION MODEL COMPARISON ==========")
print(classification_df.round(4).to_string(index=False))

classification_df.to_csv('classification_model_comparison.csv', index=False)

# Plot classification comparison
classification_df.set_index('Model')[
    ['Accuracy', 'Precision', 'Recall', 'F1 Score']
].plot(kind='bar', figsize=(11, 6))

plt.title('Classification Models Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# ==================================================
# PART 3: K-MEANS CLUSTERING
# ==================================================

X_cluster = df[['Session_Duration (hours)', 'Calories_Burned']]

cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

print("\n========== K-MEANS CLUSTERING ==========")
print("Number of records in each cluster:")
print(df['Cluster'].value_counts().sort_index())

print("\nCluster centers in original units:")
centers = cluster_scaler.inverse_transform(kmeans.cluster_centers_)

centers_df = pd.DataFrame(
    centers,
    columns=['Session_Duration (hours)', 'Calories_Burned']
)

print(centers_df.round(2))

# Plot clusters
for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]

    plt.scatter(
        cluster_data['Session_Duration (hours)'],
        cluster_data['Calories_Burned'],
        label=f'Cluster {cluster}'
    )

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=200,
    label='Cluster Centers'
)

plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.title('K-Means Clustering Results')
plt.legend()
plt.tight_layout()
plt.show()

print("\nComparison files saved successfully.")
