# Hierarchical model

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import collections as col

# import dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:-1].values

# data decomposition
pca = PCA(2)
X = pca.fit_transform(X)

# determine number of clusters using Dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Iris Clusters')
plt.ylabel('Euclidean Distance')
plt.show()

# train hierarchical model with three clusters
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising Clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], color = 'green', label='Iris-versicolor') #63
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], color = 'navy', label='Iris-setosa')      #50
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], color = 'orange', label='ris-virginica')  #37
plt.legend()
plt.title('Three Iris Clusters')
plt.show()

# Visualising cluster count
sizes = []
labels = ['Iris-versicolor', 'Iris-setosa', 'Iris-virginica']
y_cluster_count = col.Counter(y_hc)
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


