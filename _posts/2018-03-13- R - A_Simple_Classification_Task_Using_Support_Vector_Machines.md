---
title: "R : A Simple Classification Task Using Support Vector Machines"
date: 2018-03-13
tags: [Machine Learning, Classification, SVM, R]
header:
  image: "/images/img.jpg"
excerpt: "Classification, SVM, R"
---

One of the most famous areas of machine learning (ML) is Classification. It consists of identifying to which of a set of categories a new observation belongs. Classification has a broad array of applications, including ad targeting, medical diagnosis and image classification. An example would be detecting if an email is a "spam" or "non-spam".

Classification is considered an instance of supervised learning. A supervised because it is like a teacher supervising the learning process. The correct answers are known, the algorithm iteratively makes predictions on the training data and is corrected by the teacher. This consists of analyzing the training data and producing an inferred function, which can be used for mapping new examples. An optimal scenario will allow to correctly determine the class labels to which unseen instances belong.

<img src="{{ site.url }}{{ site.baseurl }}/images/A%20Simple%20Classification%20Task%20Using%20Support%20Vector%20Machines_files/lsvm.png">

In this post, we will try to classify Iris flowers using Support Vector Machines (SVM) which is a supervised learning method to analyze data and recognize patterns. It is used for both classification and regression analysis.  

The Iris flower dataset was introduced by Sir Ronald Fisher back in 1936. It consists of 50 samples from each of three species of Iris:

 * Iris Setosa
 * Iris Virginica
 * Iris Versicolor


Four features were measured from each flower: the length and the width of the sepals and petals, in centimeters.

We will use R in this post, here is the [Python version]({% post_url 2018-03-12- A_Simple_Classification_Task_Using_Support_Vector_Machines %}). So let´s start :).

First of all, we will import the necessary library:


```R
# ISLR has a collection of data-sets
# for an Introduction to Statistical Learning
# for installation: install.packages('ISLR')
library(ISLR)
```

## Getting to know the data


```R
head(iris)
```


<table>
<thead><tr><th>Sepal.Length</th><th>Sepal.Width</th><th>Petal.Length</th><th>Petal.Width</th><th>Species</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td></tr>
</tbody>
</table>




```R
summary(iris)
```


      Sepal.Length    Sepal.Width     Petal.Length    Petal.Width         Species  
     Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100      setosa:50
     1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  versicolor:50
     Median :5.800   Median :3.000   Median :4.350   Median :1.300  virginica :50
     Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
     3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
     Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  



```R
any(is.na(iris))
```



    FALSE


We can see that the dataset is clear and complete, which means we can move on to the next step.

## Exploring the Data

We will visualize data using the "ggplot2" library. The visualizations will help to get an idea about the separability across the three species.


```R
# ggplot2 is a library for creating data visualisations
# for installation: install.packages('ISLR')
library('ggplot2')
```

We already know that there are 50 instances per flower kind. So, we are sure there will be no bias factor during the analysis. If we did not know that before hand, a simple barplot can inform us about distribution in the dataset.


```R
b_plot <- ggplot(iris,aes(x=Species,fill=Species))
print(b_plot + geom_bar())
```


![png](/images/A%20Simple%20Classification%20Task%20Using%20Support%20Vector%20Machines_files/output_18_0.png)


Let´s now explore the different features and see how they contribute to Iris flowers classification.


```R
sct_pl_1 <- ggplot(iris,aes(x=Sepal.Length,y=Petal.Width,color=Species))
print(sct_pl_1 + geom_point(aes(shape=Species),size=2))
```


![png](/images/A%20Simple%20Classification%20Task%20Using%20Support%20Vector%20Machines_files/output_20_0.png)



```R
sct_pl_2 <- ggplot(iris,aes(x=Sepal.Width,y=Petal.Width,color=Species))
print(sct_pl_2 + geom_point(aes(shape=Species),size=2))
```


![png](/images/A%20Simple%20Classification%20Task%20Using%20Support%20Vector%20Machines_files/output_21_0.png)



```R
sct_pl_3 <- ggplot(iris,aes(x=Petal.Length,y=Sepal.Width,color=Species))
print(sct_pl_3 + geom_point(aes(shape=Species),size=2))
```


![png](/images/A%20Simple%20Classification%20Task%20Using%20Support%20Vector%20Machines_files/output_22_0.png)



```R
sct_pl_4 <- ggplot(iris,aes(x=Petal.Length,y=Petal.Width,color=Species))
print(sct_pl_4 + geom_point(aes(shape=Species),size=2))
```


![png](/images/A%20Simple%20Classification%20Task%20Using%20Support%20Vector%20Machines_files/output_23_0.png)


From the previous plots, it appears that the Setosa is very unique compared to the other species. It means that classifying it would be very easy. In the other hand, we can clearly see some overlapping when it comes to Versicolor and Virginica.

## Training and Testing the Data

In order to use SVM, we will load the "e1071" library.


```R
# for installation: install.packages('e1071')
library('e1071')
```


```R
# Splitting data into train & test
sample <- sample.int(n = nrow(iris), size = floor(.75*nrow(iris)), replace = F)
train <- iris[sample, ]
test  <- iris[-sample, ]
```


```R
# Training the model, for more info: help("svm")
model <- svm(Species ~ .,data=train)
```


```R
summary(model)
```



    Call:
    svm(formula = Species ~ ., data = train)


    Parameters:
       SVM-Type:  C-classification
     SVM-Kernel:  radial
           cost:  1
          gamma:  0.25

    Number of Support Vectors:  44

     ( 20 18 6 )


    Number of Classes:  3

    Levels:
     setosa versicolor virginica






```R
# Testing and Evaluating the model
predictions <- predict(model,test[1:4])
table(predictions,test[,5])
```



    predictions  setosa versicolor virginica
      setosa         13          0         0
      versicolor      1         14         2
      virginica       0          1         7


We can clearly see that the model is doing a great job in classifying the Iris flowers, especially the Setosa. For the remaining species, even thought we are predicting the correct label pretty well, we will see if we can do better by performing "Parameter Tuning".

## Tuning Parameters and Model Selection

For most ML algorithms, there are some parameters that should be adjusted to make the model more accurate.

One thing should be kept in mind though: we want a model that can be able to predict the correct label for unseen data while doing good with the used data. This means that our model should not fit training data too well (over-fitting), but also, it should not come to the point where it can neither model the training data nor generalize to new data (under-fitting).

A perfect scenario would be to select a model at the sweet spot between under-fitting and over-fitting. This is the ultimate goal of ML, but it is often very difficult to do in practice.

For now, let´s try different values for the SVM parameters. There are two parameters that could be adjusted: Cost (C) and Gamma.

The Gamma parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’.

The C parameter trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly.

Instead of trying combinations of parameters one by one and deciding which is the optimal values to choose for the model, we will use the tune() function.


```R
r_tune <- tune(svm,train.x=train[1:4],train.y=train[,5],kernel='radial',
                  ranges=list(cost=10^(-1:2), gamma=10^(-3:0)))
```


```R
summary(r_tune)
```



    Parameter tuning of ‘svm’:

    - sampling method: 10-fold cross validation

    - best parameters:
     cost gamma
       10   0.1

    - best performance: 0.009090909

    - Detailed performance results:
        cost gamma       error dispersion
    1    0.1 0.001 0.618181818 0.15681798
    2    1.0 0.001 0.481818182 0.18502365
    3   10.0 0.001 0.090909091 0.07422696
    4  100.0 0.001 0.027272727 0.04391326
    5    0.1 0.010 0.545454545 0.20814577
    6    1.0 0.010 0.081818182 0.06707862
    7   10.0 0.010 0.027272727 0.04391326
    8  100.0 0.010 0.026515152 0.04274321
    9    0.1 0.100 0.118181818 0.09630454
    10   1.0 0.100 0.027272727 0.04391326
    11  10.0 0.100 0.009090909 0.02874798
    12 100.0 0.100 0.009090909 0.02874798
    13   0.1 1.000 0.062878788 0.06103071
    14   1.0 1.000 0.009090909 0.02874798
    15  10.0 1.000 0.036363636 0.04694525
    16 100.0 1.000 0.036363636 0.04694525



The best result was achieved using the C=1 and gamma=1, let´s re-train our model with these parameters.


```R
new_model = svm(Species ~ .,data=train,kernel="radial", cost=10, gamma=0.1)
```


```R
summary(new_model)
```



    Call:
    svm(formula = Species ~ ., data = train, kernel = "radial", cost = 10,
        gamma = 0.1)


    Parameters:
       SVM-Type:  C-classification
     SVM-Kernel:  radial
           cost:  10
          gamma:  0.1

    Number of Support Vectors:  23

     ( 10 9 4 )


    Number of Classes:  3

    Levels:
     setosa versicolor virginica






```R
new_predictions <- predict(new_model,test[1:4])
table(new_predictions,test[,5])
```



    new_predictions setosa versicolor virginica
         setosa         14          0         0
         versicolor      0         13         2
         virginica       0          2         7


## Final Thoughts

We can see that we got an improvement by tuning the hyper-parameters. However, there will always be the overlap between Versicolor and Virginica which explains the misclassification. Anyways, as mentioned earlier, we would like to keep our model capability to generalize to new data in the future, so we will not go further and will settle for this last classifier.
