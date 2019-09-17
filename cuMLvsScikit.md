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

# cuML vs Scikit

In this post we will compare performance of cuMl vs scikit on following models:
- Kmeans
- linear regression
- random forest

Note: These experiments are on a single Node

```python
import numpy as np

import pandas as pd
import cudf as gd

%matplotlib inline
import matplotlib.pyplot as plt
```

## Kmeans

A lot of code is directly copied from https://github.com/rapidsai/notebooks/blob/branch-0.10/cuml/kmeans_demo.ipynb

```python
from cuml.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from sklearn.cluster import KMeans as skKMeans
from cuml.cluster import KMeans as cumlKMeans
```

```python
n_samples = 1000000
n_features = 4
```

##### Generate Data

```python
cudf_data_kmeans, cudf_labels_kmeans = make_blobs(
   n_samples=n_samples, n_features=n_features, centers=5, random_state=7)

cudf_data_kmeans = gd.DataFrame.from_gpu_matrix(cudf_data_kmeans)
cudf_labels_kmeans = gd.Series(cudf_labels_kmeans)
```

```python
scikit_data_kmeans = cudf_data_kmeans.to_pandas()
scikit_labels_kmeans = cudf_labels_kmeans.to_pandas()
```

##### Model Training

```python
%%time
#scikit kmeans model training

kmeans_sk = skKMeans(n_clusters=5,
                     n_jobs=-1)
kmeans_sk.fit(scikit_data_kmeans)
```

```python
%%time
#cuml kmeans model training


kmeans_cuml = cumlKMeans(n_clusters=5)
kmeans_cuml.fit(cudf_data_kmeans)
```

##### Comparison of results

```python
%%time
cuml_score = adjusted_rand_score(scikit_labels_kmeans, kmeans_cuml.labels_)
sk_score = adjusted_rand_score(scikit_labels_kmeans, kmeans_sk.labels_)
```

```python
threshold = 1e-4

passed = (cuml_score - sk_score) < threshold
print('compare kmeans: cuml vs sklearn labels_ are ' + ('equal' if passed else 'NOT equal'))
```

## Linear Regression

A lot of code has been copied from https://github.com/rapidsai/notebooks/blob/branch-0.10/cuml/linear_regression_demo.ipynb

```python
import os
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

from cuml.linear_model import LinearRegression as cuLR
from sklearn.linear_model import LinearRegression as skLR
```

```python
n_samples = 2**20
n_features = 399
```

##### Generate Data

```python
%%time
X,y = make_regression(n_samples=n_samples, n_features=n_features, random_state=0)

X = pd.DataFrame(X)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
```

```python
%%time
X_cudf = gd.DataFrame.from_pandas(X_train)
X_cudf_test = gd.DataFrame.from_pandas(X_test)

y_cudf = gd.Series(y_train.values)
```

##### Model Training

```python
%%time
#scikit linear regression
ols_sk = skLR(fit_intercept=True,
              normalize=True,
              n_jobs=-1)

ols_sk.fit(X_train, y_train)
```

```python
%%time
#cuml linear regression
ols_cuml = cuLR(fit_intercept=True,
                normalize=True,
                algorithm='eig')

ols_cuml.fit(X_cudf, y_cudf)
```

##### Evaluation of results

```python
%%time

#scikit evaluation
predict_sk = ols_sk.predict(X_test)

error_sk = mean_squared_error(y_test, predict_sk)
```

```python
%%time

#cuml evaluation
predict_cuml = ols_cuml.predict(X_cudf_test).to_array()

error_cuml = mean_squared_error(y_test, predict_cuml)
```

```python
print("SKL MSE(y): %s" % error_sk)
print("CUML MSE(y): %s" % error_cuml)
```

## Random Forest

Some of the code has been copied from https://github.com/rapidsai/notebooks/blob/branch-0.10/cuml/random_forest_mnmg_demo.ipynb

```python
from sklearn.metrics import accuracy_score
from sklearn import model_selection, datasets


from cuml.ensemble import RandomForestClassifier as cumlRF
from sklearn.ensemble import RandomForestClassifier as sklRF
```

```python
# Data parameters
train_size = 100000
test_size = 1000
n_samples = train_size + test_size
n_features = 20

# Random Forest building parameters
max_depth = 12
n_bins = 16
n_trees = 1000
```

##### Generate Data

```python
X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                 n_clusters_per_class=1, n_informative=int(n_features / 3),
                                 random_state=123, n_classes=5)
y = y.astype(np.int32)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)
```

```python
X_train_cudf = gd.DataFrame.from_pandas(pd.DataFrame(X_train))
y_train_cudf = gd.Series(y_train)
```

##### Model Training

```python
%%time

# Use all avilable CPU cores
skl_model = sklRF(max_depth=max_depth, n_estimators=n_trees, n_jobs=-1)
skl_model.fit(X_train, y_train)
```

```python
%%time

cuml_model = cumlRF(max_depth=max_depth, n_estimators=n_trees, n_bins=n_bins)
cuml_model.fit(X_train_cudf, y_train_cudf)
```

##### Evaluation and comparison

```python
skl_y_pred = skl_model.predict(X_test)
cuml_y_pred = cuml_model.predict(X_test)

# Due to randomness in the algorithm, you may see slight variation in accuracies
print("SKLearn accuracy:  ", accuracy_score(y_test, skl_y_pred))
print("CuML accuracy:     ", accuracy_score(y_test, cuml_y_pred))
```

```python

```
