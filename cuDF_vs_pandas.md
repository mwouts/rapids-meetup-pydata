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

# Comparing Speeds of pandas and cuDF

In this notebook, we'll explore and analyze how much faster cuDF can be compared to a traditional pandas on data manipulation operations.

The task performed here is a simple one; for various orders of magnitude (called `data_size`, ranging from 10^3 to 10^8), we will:
1. Create a single-column DataFrame with `data_size` rows of numbers `[1, 2, 3, ...]`.
2. Apply a function to all rows; doubling the number in this row.

We will compare the time it takes for both pandas DataFrame and cuDF DataFrames to complete this task.


```python
import cudf
import pandas as pd
import numpy as np
import time
```

```python
pandas_times = []
cudf_times = []

exps = range(3, 9)

for exp in exps:
    data_length = 10**exp
    print("\ndata length: 1e%d" % exp)

    df = cudf.dataframe.DataFrame()
    df['in1'] = np.arange(data_length, dtype=np.float64)


    def kernel(in1, out):
        for i, x in enumerate(in1):
            out[i] = x * 2.0

    start = time.time()
    df = df.apply_rows(kernel,
                       incols=['in1'],
                       outcols=dict(out=np.float64),
                       kwargs=dict())
    end = time.time()
    print('cuDF time', end-start)
    cudf_times.append(end-start)
    assert(np.isclose(df['in1'].sum()*2.0, df['out'].sum()))


    df = pd.DataFrame()
    df['in1'] = np.arange(data_length, dtype=np.float64)
    start = time.time()
    df['out'] = df.in1.apply(lambda x: x*2)
    end = time.time()
    print('pandas time', end-start)
    pandas_times.append(end-start)
    
    assert(np.isclose(df['in1'].sum()*2.0, df['out'].sum()))

```

```python
results = cudf.DataFrame()
results["data_size"] = ["1e%d" % exp for exp in exps]
results["pandas_time"] = pandas_times
results["cudf_time"] = cudf_times
results["speedup"] = results["pandas_time"] / results["cudf_time"]
#results = results.set_index('data_size')

#results.transpose()
results
```

```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format ='retina'

df = cudf.melt(results, id_vars=["data_size"], var_name="class", value_name="time")
df_time = df[df['class'] != "speedup"]

# We need to fall back to_pandas for plotting operations
f, ax = plt.subplots(2, 1, figsize=(8,8))
sns.barplot(x="data_size", y="time", hue="class", data=df_time.to_pandas(), ax=ax[0])
ax[0].set_yscale('log')
ax[0].set_ylabel('time (log)')
sns.lineplot(x="data_size", y="speedup", data=results.to_pandas(), ax=ax[1], color='g')
ax[1].yaxis.grid()
```

# Takeaways

From the graph above, we can clearly see the choice you'll have to make when deciding to switch to GPU for your operations.

For relatively small size of data, the cost of moving this array into GPU memory is not worth the speed of processing that GPU offer, and you lose time. However, as data becomes bigger (10e6 and up), the parallelization capabilities of the GPU becomes really powerful and can speed up the operation up to **120** !
