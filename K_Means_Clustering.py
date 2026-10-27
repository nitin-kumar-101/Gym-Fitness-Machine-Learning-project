
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# Step 2: Select features for clustering
X = df[['Session_Duration (hours)', 'Calories_Burned']]

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Step 5: Train the model and assign clusters
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Display cluster information
print("K-MEANS CLUSTERING RESULTS")
print("--------------------------")
print("\nNumber of records in each cluster:")
print(df['Cluster'].value_counts().sort_index())

print("\nCluster Centers (original units):")
centers = scaler.inverse_transform(kmeans.cluster_centers_)
print(pd.DataFrame(
    centers,
    columns=['Session_Duration (hours)', 'Calories_Burned']
).round(2))

# Step 7: Visualize clusters
plt.figure(figsize=(9, 6))

for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]

    plt.scatter(
        cluster_data['Session_Duration (hours)'],
        cluster_data['Calories_Burned'],
        label=f'Cluster {cluster}',
        s=35
    )

# Plot cluster centers
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=200,
    edgecolors='black',
    label='Cluster Centers'
)

plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.title('K-Means Clustering of Workout Records')
plt.legend()
plt.show()

