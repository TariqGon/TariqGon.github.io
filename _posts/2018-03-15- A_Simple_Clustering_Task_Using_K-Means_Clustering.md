---
title: "Python : An Unsupervised Learning Task Using K-Means Clustering"
date: 2018-03-15
tags: [Machine Learning,Data Science,Clustering, K-Means Clustering, Python]
header:
  image: "/images/img.jpg"
excerpt: " Data Science, Clustering, K-Means Clustering, Python"
---

In the previous post, we performed a supervised machine learning in order to classify Iris flowers, and did pretty well in predicting the labels (kinds) of flowers. We could evaluate the performance of our model because we had the "species" column with the name of three iris kinds. Now, let´s imagine that we were not given this column and that we wanted to know if there are different kinds of Iris flower based only on the measurements of length and the width of the sepals and petals. Well, this is a called an unsupervised learning.

Unsupervised learning means that there is no class (label) column which we can use to test and evaluate how well a model is performing. So there is no outcome to be predicted, therefore the goal is trying to find patterns in the data to reach a reasonable conclusion.

We will use the K-means clustering algorithm on our Iris data assuming that we do not have the "species" column. We will investigate if data can be grouped into 3 clusters representing the three species of Iris (Iris setosa, Iris virginica and Iris versicolor).


We will use Python in this post, here is the [R version]({% post_url 2018-03-14- R - A_Simple_Clustering_Task_Using_K-Means_Clustering %}). So let´s dive in :).

## Preparing the data


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
iris = sns.load_dataset("iris")
data = iris.drop("species",1)
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



## Exploring the Data

Let´s now do some exploratory data analysis to see if we can get an idea about the data. Remember, we assume we don´t know anything about how many clusters (kinds) of flowers we have from the dataset.


```python
sns.lmplot(x='sepal_length',y='petal_width',data=data,fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x7fcf98ac0320>




![png](/images/KMeansClustering/output_11_1.png)



```python
sns.lmplot(x='sepal_width',y='petal_width',data=data,fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x7fcf98adf4a8>




![png](/images/KMeansClustering/output_12_1.png)



```python
sns.lmplot(x='petal_length',y='sepal_width',data=data,fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x7fcf90c93550>




![png](/images/KMeansClustering/output_13_1.png)



```python
sns.lmplot(x='petal_length',y='petal_width',data=iris,fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x7fcf90b99d30>




![png](/images/KMeansClustering/output_14_1.png)


From the previous plots, it seems that the data can be grouped into, minimum, two clusters. One cluster will definitely be easy for K-means to determine, while the others might get tricky to define.

The K-means clustering algorithm randomly assigns each observation to a cluster, and finds the centroid of each cluster.

The next process consists of iterating through two steps till the within cluster variation cannot be reduced any further.:

* Reassign data points to the cluster whose centroid is closest.
* Calculate new centroid of each cluster.

Since this is an unsupervised task, there is no training or testing step, we will go on and try different clustering numbers for our data and visualize the result.

## K-Means Clustering


```python
# Importing KMeans from slLearn
from sklearn.cluster import KMeans
```


```python
# Create an instance of a K Means model with 2 clusters
# Since we are supposed to not know there are 3 species
kmeans_two_md = KMeans(n_clusters=2)
```


```python
# Fitting the model
kmeans_two_md.fit(data)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
# Cluster center vectors
kmeans_two_md.cluster_centers_
```




    array([[ 5.00566038,  3.36981132,  1.56037736,  0.29056604],
           [ 6.30103093,  2.88659794,  4.95876289,  1.69587629]])




```python
plt.scatter(x='petal_length',y='sepal_width',data=data,c=kmeans_two_md.labels_)
```




    <matplotlib.collections.PathCollection at 0x7fcf8b31a278>




![png](/images/KMeansClustering/output_24_1.png)



```python
# Create an instance of a K Means model with 3 clusters
kmeans_three_md = KMeans(n_clusters=3)
```


```python
# Fitting the model
kmeans_three_md.fit(data)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)




```python
# Cluster center vectors
kmeans_three_md.cluster_centers_
```




    array([[ 6.85      ,  3.07368421,  5.74210526,  2.07105263],
           [ 5.006     ,  3.428     ,  1.462     ,  0.246     ],
           [ 5.9016129 ,  2.7483871 ,  4.39354839,  1.43387097]])




```python
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(12,6))
iris_map = {'virginica':1,'setosa':2, 'versicolor':3}
ax1.set_title('K Means')
ax1.scatter(x='petal_length',y='sepal_width',data=data,c=kmeans_three_md.labels_)
ax2.set_title("Original")
ax2.scatter(x='petal_length',y='sepal_width',data=iris,c=iris['species'].apply(lambda x: iris_map[x]))
```




    <matplotlib.collections.PathCollection at 0x7fcf8ae20588>




![png](/images/KMeansClustering/output_28_1.png)


As expected, the Setosa was correctly grouped, however, as seen in the previous post, there will always be that overlap between Versicolor and Virginica which explains the misclassification.
