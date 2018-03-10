---
date: 2018-03-10
tags: [machine learning,data science,data preprocessing]
header:
  image: ""
excerpt: " Data Science, Data Preprocessing"
mathjax: "true"
---

An entry point to Data Analysis can be a regression example (Next Post). But first things first, data should be prepared before the analysis can be performed. This step is called preprocessing. Real-world (raw) data can be inconsistent or incomplete and can even contain errors. Through the following lines, we will try to walk through a simple data preprocessing task using a famous dataset.

A statistician with the name Francis Galton wanted to see if there was a connection between the height of sons and the height of their fathers. He measured the height of fathers and sons across 205 families.

We will look closely at the data he used ( which can be found  [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/T0HSJ1) ) , explore it and see what different information it contains and do some data preprocessing.

We will use Python in this post, so let´s dive in :).

First of all we will import the necessary packages:


```python
# Pandas is an open souce library for data manipulation and analysis
# NumPy is the fundamental package for scientific computing
import pandas as pd
import numpy as np
```

We can read the data into a Dataframe which is a data structure with columns of potentially different types. It basically  looks like a spreadsheet or SQL table. And since we have a tab-separated data file (data separated with a tab rather than a comma in a csv(comma-separated values)), We need to specify it as follows:


```python
data = pd.read_csv("galton-data.tab", sep="\t")
```

Now that we have our data in a Dataframe, we can use head() to see the first 5 rows.


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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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


```python
# How many rows and columns are in the dataframe ?
data.shape
```




    (898, 8)




```python
# What are the available data concerning family number 7
data.loc[data["family"] == "7"]
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
      <th>22</th>
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
      <th>23</th>
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
      <th>24</th>
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
      <td>F</td>
      <td>70.5</td>
      <td>6</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>27</th>
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




```python
# what is the maximum height of fathers
data["father"].max()
```




    78.5




```python
# what is the maximum height of sons
data.loc[data["male"] == 1,"height"].max()
```




    79.0




```python
# what is the maximum height of mothers
data["mother"].max()
```




    70.5




```python
# what is the maximum height of daughters
data.loc[data["female"] == 1,"height"].max()
```




    70.5




```python
# what is the mean height of fathers
data["father"].mean()
```




    69.23285077950997




```python
# what is the mean height of sons
data.loc[data["male"] == 1,"height"].mean()
```




    69.22881720430114




```python
# What is the number of observed families
data["family"].nunique()
```




    197



From the study of Francis Galton we know that he gathered data across 205 families. This means that we have some missing data.

A part of data analysis is dealing with missing or incomplete data. We will try to take a look into the last rows of our dataframe just out of curiosity. For this we will use tail() and specify that we want the last 20 rows


```python
data.tail(20)
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
      <th>878</th>
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
      <th>879</th>
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
      <td>60.0</td>
      <td>7</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>882</th>
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
      <th>883</th>
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
      <th>884</th>
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
      <th>885</th>
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
      <th>886</th>
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
      <th>887</th>
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
      <th>888</th>
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
      <th>889</th>
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
      <th>890</th>
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
      <th>891</th>
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
      <th>892</th>
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
      <th>893</th>
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
      <th>894</th>
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
      <th>895</th>
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
      <th>896</th>
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
      <th>897</th>
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


```python
# What is the data type of each column? (numerical or character values?)
data.dtypes
```




    family     object
    father    float64
    mother    float64
    gender     object
    height    float64
    kids        int64
    male      float64
    female    float64
    dtype: object



We notice that each of the height columns : "father", "mother", "height"(of the adult children) are float64, a.k.a. floating-point number. For the column: "kid" as it indicates the number of children it should be obviously an integer.  

Let´s look now at the remaining two culumns with the data type: "object".

First, the column "gender" is supposed to have one of two values: F for female, or M for male. let´s make sure that it is the case.


```python
data["gender"].unique()
```




    array(['M', 'F'], dtype=object)



Out of curiosity, we want to see the number of both of them.


```python
data["gender"].value_counts()
```




    M    465
    F    433
    Name: gender, dtype: int64



Data like gender, blood types, ratings are usually called a categorical data, because they can take on only a limited, and usually fixed, number of possible values. So, we can convert it from "object" data type to "category".


```python
data["gender"] = data["gender"].astype("category")
```


```python
data["gender"].unique()
```




    [M, F]
    Categories (2, object): [M, F]



The last one is "family". This is the column which  can be considered as a family identification number. It should be an integer as well. But, since it has the "136A" value, it was given the type "object".

We want to check this column for any missing values:


```python
familynb = list(data.family.unique())
seq = map(str, range(1,206))
```


```python
set(seq) - set(familynb)
```




    {'111', '120', '13', '161', '189', '202', '205', '50', '84'}



Well, we can see that we have 9 values missing. If we include the "136A" family, we should have data for 197, which is the number we found earlier.

We will see if the "136A" is the same as the family number 136, to verify it was not a mistyping case.


```python
data.loc[data["family"] == "136"]
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
      <th>588</th>
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
      <th>589</th>
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
      <td>67.0</td>
      <td>10</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>592</th>
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
      <th>593</th>
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
      <th>594</th>
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
      <td>62.0</td>
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
      <td>61.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.loc[data["family"] == "136A"]
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
      <th>890</th>
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
      <th>891</th>
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
      <th>892</th>
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
      <th>893</th>
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
      <th>894</th>
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
      <th>895</th>
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
      <th>896</th>
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
      <th>897</th>
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


```python
data.loc[data["family"] == "136A","family"] = "205"
```

We can check the modification to be sure everything is in order


```python
data.loc[data["family"] == "205"]
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
      <th>890</th>
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
      <th>891</th>
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
      <th>892</th>
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
      <th>893</th>
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
      <th>894</th>
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
      <th>895</th>
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
      <th>896</th>
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
      <th>897</th>
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


```python
data["family"] = data["family"].astype("int")
```


```python
# We can check data types again
data.dtypes
```




    family       int64
    father     float64
    mother     float64
    gender    category
    height     float64
    kids         int64
    male       float64
    female     float64
    dtype: object



As a last step, we should make sure there is no missing data a.k.a. NaN value. this can be done using the following code:


```python
data.isnull().values.any()
```




    False
