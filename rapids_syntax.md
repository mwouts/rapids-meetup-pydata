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

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns


import cudf
import pandas as pd
print("cuDF version:", cudf.__version__)
print("pandas version:", pd.__version__)

%matplotlib inline
%config InlineBackend.figure_format ='retina'
```

# Create DataFrame

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
pdf = pd.DataFrame({'a': [0, 1, 2, 3],'b': [0.1, 0.2, None, 0.3]})
gdf = cudf.from_pandas(pdf)

gdf
```

```python
# Read from CSV
gdf = cudf.read_csv('diabetes.csv', delimiter=',')

gdf.head(3)
```

# Data Operation

```python
# Some functions are not implemented; we need to go back to pandas.
gdf.to_pandas().info()
```

```python
gdf.describe()
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

```python

```

```python

```

```python

```

```python

```
