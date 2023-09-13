# K-means model

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import collections as col

# import dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:-1].values

# Data Decomposition
pca = PCA(2)
X = pca.fit_transform(X)

# determine number of clusters
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Elbow graph
plt.plot(range(1, 10), wcss, color = 'blue')
plt.title('Elbow Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()    

# train kmeans model with 3 cluaters
kmeans = KMeans(n_clusters=3, init='k-means++')
y_kmeans = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_

# 2D Visualising clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'green', label='Iris-versicolor') #61
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'navy', label='Iris-setosa') #50
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'orange', label='Iris-virginica') #39
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='s', color='blue', label='Centroids')
plt.legend()
plt.title('Three Iris Clusters')
plt.show()

# Visualising cluster count
sizes = []
labels = ['Iris-versicolor', 'Iris-setosa', 'Iris-virginica']
y_cluster_count = col.Counter(y_kmeans)
for key, value in y_cluster_count.items():
    plt.bar(key, value, width=0.3, label = labels[key])
    plt.text(key, value-3, str(value))
    sizes.append(value)
plt.title('Cluster Items Counts')
plt.xlabel('Clusters')
plt.ylabel('Count')
plt.legend()
plt.show()


# pie chart
sizes.sort()
sizes.reverse()
plt.pie(sizes, labels= labels, autopct='%1.1f%%', colors=[ 'green', 'navy', 'orange'])
plt.title('Cluster Percentages')
plt.show()

