import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('mental.csv')


categorical_columns = ['posts', 'predicted']
numerical_columns = ['intensity']

encoder = OneHotEncoder() 
transformed_categorical = encoder.fit_transform(data[categorical_columns])

transformed_data = tf.constant(tf.concat([transformed_categorical, data[numerical_columns].values], axis=1), dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices(transformed_data)

num_clusters = 5
num_iterations = 100

initial_clusters = tf.random.shuffle(transformed_data).take(num_clusters)

kmeans = tf.raw_ops.KMeans(
    data=transformed_data,
    k=num_clusters,
    iterations=num_iterations,
    initial_clusters=initial_clusters,
    distance_metric='EUCLIDEAN'
)
cluster_centers = kmeans.cluster_centers.numpy()
cluster_indices = kmeans.cluster_indices.numpy()

plt.figure(figsize=(8, 6))

x = transformed_data[:, 0].numpy()  
y = transformed_data[:, 1].numpy() 

plt.scatter(x, y, c=cluster_indices, cmap='viridis', alpha=0.5)

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', color='red', s=100, label='Cluster Centers')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')

plt.legend()
plt.grid(True)
plt.show()
