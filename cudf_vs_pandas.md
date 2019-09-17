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
df = cudf.melt(results, id_vars=["data_size"], var_name="class", value_name="time")
df
```

```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format ='retina'
df_time = df[df['class'] != "speedup"]

f, ax = plt.subplots(2, 1, figsize=(8,8))
sns.barplot(x="data_size", y="time", hue="class", data=df_time.to_pandas(), ax=ax[0])
ax[0].set_yscale('log')
ax[0].set_ylabel('time (log)')
sns.lineplot(x="data_size", y="speedup", data=results.to_pandas(), ax=ax[1], color='g')
ax[1].yaxis.grid()
```

```python

```
