---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# cuDF syntax examples

In this notebook, we will give some examples on how similar the cuDF syntax is to the pandas syntax.

For plots and column operations, we are using the [Pima Indians Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database/), found on Kaggle, and stored in `data/diabetes.csv`.

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns


import cudf
import pandas as pd

# Let's print the cuDF and pandas versions, newer updates might change the conclusions of this notebook
print("cuDF version:", cudf.__version__)
print("pandas version:", pd.__version__)

%matplotlib inline
%config InlineBackend.figure_format ='retina'
```

# Create DataFrame

DataFrame creation operations are pretty similar to the pandas API, you can even load data from an existing pandas DataFrame.

Like pandas, with Jupyter notebooks there is a nice table-like output to visualize your DataFrame.

```python
# Define by column
gdf = cudf.DataFrame()
gdf['my_column'] = [1, 2, 3]

gdf
```

```python
# Define from dict
gdf = cudf.DataFrame({'a': [1, 2, 3],
                      'b': [4, 5, 6]})

gdf
```

```python
# Load a pandas dataframe in GPU
pdf = pd.DataFrame({'a': [0, 1, 2, 3],'b': [0.1, 0.2, None, 0.3]})
gdf = cudf.from_pandas(pdf)

gdf
```

```python
# Read from CSV
gdf = cudf.read_csv('data/diabetes.csv', delimiter=',')

gdf.head(3)
```

<!-- #region -->
# Data Operations
After loading the data, you want to get some information, this step is called EDA (for Exploratory Data Analysis). 

Here, not all pandas functions that you usually use are (yet) implemented in cuDF. Take a look at [the official cuDF documentation]() for the exact list of existing methods. In summary, you can:

-   Do some dataset operations, such as `describe`
-   Get means, std, unique values of columns
-   Filter rows / group by / sort values
-   Apply user-defined functions to every row if your DataFrame
-   Merge, concatenate DataFrames

The takeaway here is that if you try something you're used to in pandas, but in cuDF, and it doesn't work, you can easily get back to the pandas API by writing:
```
    my_cudf_dataframe.to_pandas()
```
---
<!-- #endregion -->

```python
gdf.describe()
```

```python
# Some functions are not implemented; we need to go back to pandas.
gdf.to_pandas().info()
```

```python
# Plotting using seaborn
ax = sns.distplot(gdf['Age'])

# Some seaborn plots require us to fall back to pandas
plt.figure(figsize=(14,4))
ax = sns.barplot(x='Age', y='Pregnancies', data=gdf.to_pandas())
```

```python
# Average, Std of columns
print("Age avg:", gdf['Age'].mean())
print("Age std:", gdf['Age'].std())
```

```python
# Some DataFrame operations are not (yet) implemented and we need to fall back to pandas

# Example correlation matrix
corr = gdf.to_pandas().corr()

f, ax = plt.subplots(figsize=(9, 7))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, linewidths=1, cbar_kws={"shrink": .5})
```

```python
# Value counts
gdf['Outcome'].value_counts()
```

```python
# Number of unique values
gdf['Age'].nunique()
```

```python
gdf.loc[2:3, ['Age', 'Glucose']]
```

```python
# Filter data
gdf.query('Age > 65')
```

```python
gdf.sort_values(by='Age', ascending=False).head(10)
```

```python
# Do group by operations
gdf.groupby(['Age']).agg({'Pregnancies': 'sum'}).head()
```

```python
# Apply operation to a Series
def double_age(age):
    return age * 2
gdf['Age'].astype(np.float64)
gdf['Age_doubled'] = gdf['Age'].applymap(lambda x: x*2)
gdf.head(10)
```

```python
# Apply operation to a DataFrame
def triple_age(Age, Age_tripled):
    for i, age in enumerate(Age):
        Age_tripled[i] = age * 3.0 
    
gdf.apply_rows(triple_age,
               incols=['Age'],
               outcols=dict(Age_tripled=np.int),
               kwargs=dict()
              ).head(10)
```
