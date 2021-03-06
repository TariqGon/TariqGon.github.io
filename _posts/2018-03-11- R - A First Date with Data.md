---
title: "R : A First Date with Data"
date: 2018-03-11
tags: [Machine Learning, Data Preprocessing, R]
header:
  image: "/images/img.jpg"
excerpt: "Data Preprocessing, R"
---

An entry point to Data Analysis can be a ([Regression Example](/_pages/2018-03-11- A_Simple_Linear_Regression_Example.html)). But first things first, data should be prepared before the analysis can be performed. This step is called preprocessing. Real-world (raw) data can be inconsistent or incomplete and can even contain errors. Through the following lines, we will try to walk through a simple data preprocessing task using a famous dataset.

A statistician with the name Francis Galton wanted to see if there was a connection between the heights of sons and their fathers. He measured the height of children and their parents across 205 families.

We will look closely at the data he used (which can be found  [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/T0HSJ1)) , explore it and see what different information it contains and do some data preprocessing.

We will use R in this post, here is the [Python version]({% post_url 2018-03-10- A_First_Date_ with_Data %}). So let´s dive in :).

We can read the data into a Dataframe which is a data structure with columns of potentially different types. It basically  looks like a spreadsheet or SQL table. And since we have a tab-separated data file (data separated with a tab rather than a comma in "comma-separated values"), We need to specify it as follows:


```R
df <- read.table("galton-data.tab", sep = '\t',header = TRUE)
```

Now that we have our data in a Dataframe, we can use head() to see the first 5 rows.


```R
head(df)
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
      <th>family</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>height</th>
      <th>kids</th>
      <th>male</th>
      <th>female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>78.5</td>
      <td>67.0</td>
      <td>M</td>
      <td>73.2</td>
      <td>4</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>78.5</td>
      <td>67.0</td>
      <td>F</td>
      <td>69.2</td>
      <td>4</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>78.5</td>
      <td>67.0</td>
      <td>F</td>
      <td>69.0</td>
      <td>4</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>78.5</td>
      <td>67.0</td>
      <td>F</td>
      <td>69.0</td>
      <td>4</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>75.5</td>
      <td>66.5</td>
      <td>M</td>
      <td>73.5</td>
      <td>4</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


We can see that there are 8 columns. For each of the adult children of one family, we have the data about their height and gender as well as their parents height and the number of siblings.

Now, we will try to find more about our DataFrame by asking some basic questions:


```R
# # How many rows and columns are in the dataframe ?
dim(df)
```




    (898, 8)




```R
# What are the available data concerning family number 7
df[df$family == "7",]
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
      <th>family</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>height</th>
      <th>kids</th>
      <th>male</th>
      <th>female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>7</td>
      <td>74.0</td>
      <td>68.0</td>
      <td>M</td>
      <td>76.5</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>7</td>
      <td>74.0</td>
      <td>68.0</td>
      <td>M</td>
      <td>74.0</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>7</td>
      <td>74.0</td>
      <td>68.0</td>
      <td>M</td>
      <td>73.0</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>7</td>
      <td>74.0</td>
      <td>68.0</td>
      <td>M</td>
      <td>73.0</td>
      <td>6</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>7</td>
      <td>74.0</td>
      <td>68.0</td>
      <td>F</td>
      <td>70.5</td>
      <td>6</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7</td>
      <td>74.0</td>
      <td>68.0</td>
      <td>F</td>
      <td>64.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


```R
# what is the maximum height of fathers
max(df$father)
```




    78.5




```R
# what is the maximum height of sons
max(df[df$male == "1","height"])
```




    79




```R
# what is the maximum height of mothers
max(df$mother)
```




    70.5




```R
# what is the maximum height of daughters
max(df[df$female == "1","height"])
```




    70.5




```R
# what is the mean height of fathers
mean(df$father)
```




    69.23285077951





```R
# what is the mean height of sons
mean(df[df$male == "1","height"])
```




    69.2288172043011





```R
# What is the number of observed families
length(unique(df$family))
```




    197




From the study of Francis Galton we know that he gathered data across 205 families. This means that we have some missing data.

A part of data analysis is dealing with missing or incomplete data. We will try to take a look into the last rows of our dataframe just out of curiosity. For this we will use tail() and specify that we want the last 20 rows


```R
tail(df,20)
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
      <th>family</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>height</th>
      <th>kids</th>
      <th>male</th>
      <th>female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>879</th>
      <td>199</td>
      <td>64.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>65.0</td>
      <td>7</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>880</th>
      <td>199</td>
      <td>64.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>64.0</td>
      <td>7</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>881</th>
      <td>199</td>
      <td>64.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>64.0</td>
      <td>7</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>882</th>
      <td>199</td>
      <td>64.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>60.0</td>
      <td>7</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>883</th>
      <td>200</td>
      <td>64.0</td>
      <td>63.0</td>
      <td>M</td>
      <td>64.5</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>884</th>
      <td>201</td>
      <td>64.0</td>
      <td>60.0</td>
      <td>M</td>
      <td>66.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>885</th>
      <td>201</td>
      <td>64.0</td>
      <td>60.0</td>
      <td>F</td>
      <td>60.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>203</td>
      <td>62.0</td>
      <td>66.0</td>
      <td>M</td>
      <td>64.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>203</td>
      <td>62.0</td>
      <td>66.0</td>
      <td>F</td>
      <td>62.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>203</td>
      <td>62.0</td>
      <td>66.0</td>
      <td>F</td>
      <td>61.0</td>
      <td>3</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>204</td>
      <td>62.5</td>
      <td>63.0</td>
      <td>M</td>
      <td>66.5</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>204</td>
      <td>62.5</td>
      <td>63.0</td>
      <td>F</td>
      <td>57.0</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>891</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>72.0</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>892</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>70.5</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>893</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>68.7</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>894</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>68.5</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>895</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>67.7</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>896</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>64.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>897</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>63.5</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>898</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>63.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


With a quick look into the family column we find that the number 202 is missing from the count and, as a bonus, there appears to be a strange value "136A" in the end of our dataframe.


```R
# What is the data type of each column? (numerical or character values?)
str(df)
```

    'data.frame':	898 obs. of  8 variables:
     $ family: Factor w/ 197 levels "1","10","100",..: 1 1 1 1 108 108 108 108 123 123 ...
     $ father: num  78.5 78.5 78.5 78.5 75.5 75.5 75.5 75.5 75 75 ...
     $ mother: num  67 67 67 67 66.5 66.5 66.5 66.5 64 64 ...
     $ gender: Factor w/ 2 levels "F","M": 2 1 1 1 2 2 1 1 2 1 ...
     $ height: num  73.2 69.2 69 69 73.5 72.5 65.5 65.5 71 68 ...
     $ kids  : int  4 4 4 4 4 4 4 4 2 2 ...
     $ male  : num  1 0 0 0 1 1 0 0 1 0 ...
     $ female: num  0 1 1 1 0 0 1 1 0 1 ...


We notice that each of the height columns : "father", "mother", "height"(of the adult children) are numeric values. For the column: "kid" as it indicates the number of children it should be obviously an integer.

Let´s look now at the remaining two culumns with the data type: "Factor".

Data like gender, blood types, ratings are usually called a categorical data, because they can take on only a limited, and usually fixed, number of possible values. So, in R, they are called "Factor" columns.

First, the column "gender" is supposed to have one of two values: F for female, or M for male. let´s make sure that it is the case.


```R
unique(df$gender)
```


      M   F



The last one is "family". This is the column which  can be considered as a family identification number. It should be an integer as well. But, since it has the "136A" value, it was given the type "Factor".

We want to check this column for any missing values:


```R
fml_id <- unique(df$family)
```


```R
nb_id <- as.factor(1:205)
```


```R
setdiff(levels(nb_id), levels(fml_id))
```



	'13'  '50'  '84'  '111'  '120'  '161'  '189'  '202'  '205'




Well, we can see that we have 9 values missing. If we include the "136A" family, we should have data for 197, which is the number we found earlier.

We will see if the "136A" is the same as the family number 136, to verify it was not a mistyping case.


```R
df[df$family == "136",]
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
      <th>family</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>height</th>
      <th>kids</th>
      <th>male</th>
      <th>female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>589</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>M</td>
      <td>71.0</td>
      <td>10</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>590</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>M</td>
      <td>68.0</td>
      <td>10</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>591</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>M</td>
      <td>68.0</td>
      <td>10</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>592</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>M</td>
      <td>67.0</td>
      <td>10</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>593</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>65.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>594</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>64.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>595</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>63.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>596</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>63.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>597</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>62.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>598</th>
      <td>136</td>
      <td>68.0</td>
      <td>64.0</td>
      <td>F</td>
      <td>61.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


```R
df[df$family == "136A",]
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
      <th>family</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>height</th>
      <th>kids</th>
      <th>male</th>
      <th>female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>891</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>72.0</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>892</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>70.5</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>893</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>68.7</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>894</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>68.5</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>895</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>67.7</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>896</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>64.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>897</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>63.5</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>898</th>
      <td>136A</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>63.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


Well, we can clearly see that there are two different families. it might be a good idea to replce the "136A" with an acual number. We can choose the number "205" since it is next on the list.


```R
levels(df$family) <- c(levels(df$family), "205")
df$family[df$family == "136A"] <- "205"
```


```R
df[df$family == "205",]
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
      <th>family</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>height</th>
      <th>kids</th>
      <th>male</th>
      <th>female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>891</th>
      <td>205</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>72.0</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>892</th>
      <td>205</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>70.5</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>893</th>
      <td>205</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>68.7</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>894</th>
      <td>205</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>68.5</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>895</th>
      <td>205</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>M</td>
      <td>67.7</td>
      <td>8</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>896</th>
      <td>205</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>64.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>897</th>
      <td>205</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>63.5</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>898</th>
      <td>205</td>
      <td>68.5</td>
      <td>65.0</td>
      <td>F</td>
      <td>63.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


Now that we are sure every value in "family" is numeric, we can convert it to be so, as follows:


```R
df$family <- as.numeric(as.character(df$family))
```


```R
# We can check the change we made as follows
str(df)
```

    'data.frame':	898 obs. of  8 variables:
     $ family: num  1 1 1 1 2 2 2 2 3 3 ...
     $ father: num  78.5 78.5 78.5 78.5 75.5 75.5 75.5 75.5 75 75 ...
     $ mother: num  67 67 67 67 66.5 66.5 66.5 66.5 64 64 ...
     $ gender: Factor w/ 2 levels "F","M": 2 1 1 1 2 2 1 1 2 1 ...
     $ height: num  73.2 69.2 69 69 73.5 72.5 65.5 65.5 71 68 ...
     $ kids  : int  4 4 4 4 4 4 4 4 2 2 ...
     $ male  : num  1 0 0 0 1 1 0 0 1 0 ...
     $ female: num  0 1 1 1 0 0 1 1 0 1 ...


As a last step, we should make sure there is no missing data, a.k.a NA values.


```R
any(is.na(df))
```




    FALSE
