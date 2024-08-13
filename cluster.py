import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('mental.csv')

# Separate categorical and numerical columns
categorical_columns = ['posts', 'predicted']
numerical_columns = ['intensity']

# One-hot encode categorical columns
encoder = OneHotEncoder()  # Use sparse=False to ensure a dense array
transformed_categorical = encoder.fit_transform(data[categorical_columns])

# Combine numerical and transformed categorical data
transformed_data = tf.constant(tf.concat([transformed_categorical, data[numerical_columns].values], axis=1), dtype=tf.float32)

# Convert to TensorFlow dataset (optional, depending on usage)
dataset = tf.data.Dataset.from_tensor_slices(transformed_data)

# K-means clustering parameters
num_clusters = 5
num_iterations = 100

# Randomly initialize cluster centers
initial_clusters = tf.random.shuffle(transformed_data).take(num_clusters)

# Perform K-means clustering
kmeans = tf.raw_ops.KMeans(
    data=transformed_data,
    k=num_clusters,
    iterations=num_iterations,
    initial_clusters=initial_clusters,
    distance_metric='EUCLIDEAN'
)

# Get cluster centers and indices
cluster_centers = kmeans.cluster_centers.numpy()
cluster_indices = kmeans.cluster_indices.numpy()

# Plotting the clustered data and cluster centers
plt.figure(figsize=(8, 6))

# Extract x and y coordinates from transformed_data (assuming 2D)
x = transformed_data[:, 0].numpy()  # Adjust indices as per your dataset
y = transformed_data[:, 1].numpy()  # Adjust indices as per your dataset

# Plot original data points with cluster colors
plt.scatter(x, y, c=cluster_indices, cmap='viridis', alpha=0.5)

# Plot cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=100, label='Cluster Centers')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')

plt.legend()
plt.grid(True)
plt.show()
