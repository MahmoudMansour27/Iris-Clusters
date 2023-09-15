# Machine Learning Models: K-Means Clustering and Hierarchical Clustering

## Introduction

Welcome to the Machine Learning Clustering Models repository! This repository contains Python implementations of two powerful clustering algorithms: K-Means Clustering and Hierarchical Clustering. Clustering is an essential unsupervised learning technique used for grouping similar data points together, and these models provide insights into different approaches for achieving this goal.

## Iris Dataset

**Dataset Details:**

- **Number of Instances:** 150
- **Number of Attributes:** 5 (Columns)
- **Attribute Information:**
  1. Sepal Length (in centimeters)
  2. Sepal Width (in centimeters)
  3. Petal Length (in centimeters)
  4. Petal Width (in centimeters)
  5. Species
     - Setosa
     - Versicolor
     - Virginica

## Prerequisites

Before you can use this model, make sure you have the following prerequisites installed:

- Python > 3.x
- Pandas
- scikit-learn
- Matplotlib (for visualization, optional)
- Scipy
- Collections

```bash
pip install pandas scikit-learn matplotlib scipy collections
```

## K-Means Clustering

**Overview:**

K-Means clustering is a popular unsupervised machine learning algorithm used for partitioning a dataset into groups, or clusters, based on similarity. It aims to group data points into K clusters, with each cluster having data points that are closer to each other in terms of a chosen distance metric. K-Means is widely employed in various applications, such as customer segmentation, image compression, and anomaly detection.

**Key Concepts:**

- **Centroid-Based Clustering:** In K-Means, clusters are represented by their centroids, which are the mean positions of the data points within each cluster.

- **Distance Metric:** The algorithm typically uses Euclidean distance to measure the similarity between data points and centroids, but other distance metrics can be employed based on the data's nature.

- **Iterative Process:** K-Means follows an iterative approach, where it assigns data points to the nearest cluster centroid, updates the centroids based on the new assignments, and repeats these steps until convergence.

- **Choosing K:** Selecting the optimal number of clusters, K, is a critical step. Common methods for this include the Elbow Method and Silhouette Score. 

## Data Visualisation

1. The elbow graph is a visual representation used in K-Means clustering to determine the optimal number of clusters:

   <img src="./img/elbow graph kmeans.png" title="" alt="" width="442">

3. A scatter plot 2D for composed data:

   <img src="./img/scatter kmeans.png" title="" alt="" width="442">
  
5. Bar Chart for representing the size of each cluster:

   <img src="./img/bar kmeans.png" title="" alt="" width="442">
   
7. Pie Chart for the percentage of each cluster:

   <img src="./img/pie kmeans.png" title="" alt="" width="442">

---

## Hierarchical Clustering

**Overview:**

Hierarchical clustering is another unsupervised machine learning algorithm used to group data points into a hierarchy of clusters. Unlike K-Means, hierarchical clustering doesn't require specifying the number of clusters in advance. It creates a tree-like structure of clusters, known as a dendrogram, which can be cut at different levels to obtain different numbers of clusters. Hierarchical clustering is commonly used in biology, taxonomy, and document clustering.

**Key Concepts:**

- **Agglomerative:** Starts with individual data points as clusters and merges them iteratively until one cluster remains.

- **Dendrogram:** A graphical representation of the hierarchical structure of clusters, where the vertical lines represent clusters and their heights represent the dissimilarity between merged clusters.

- **Linkage Methods:** Different linkage methods, such as single, complete, and average linkage, determine how the dissimilarity between clusters is calculated during the merging process.

## Data Visualisation

1. A dendrogram is a tree-like diagram. It displays the hierarchical relationships between data points or clusters and used to determine optimal number of clusters:

   <img src="./img/Dendrogram hc.png" title="" alt="" width="442">
   
3. A scatter plot 2D for composed data:

   <img src="./img/scatter hc.png" title="" alt="" width="442">

5. Bar Chart for representing the size of each cluster:

   <img src="./img/bar hc.png" title="" alt="" width="442">

7. Pie Chart for the percentage of each cluster:

   <img src="./img/pie hc.png" title="" alt="" width="442">

## Conclusion

In conclusion, both the K-Means clustering and Hierarchical clustering models offer valuable insights into grouping data points based on similarity, albeit through different approaches. Despite their distinct methodologies, it's noteworthy that the results from these two models often exhibit remarkable similarity.
