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

## Pytorch Encoder for cudf vs Pandas

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cudf as gd
```

## Reading and Modifying Data

We are using the Electrolysia Time-Series Electricity Consumption Dataset from Kaggle, you can find more details about it [on the Kaggle dataset page](https://www.kaggle.com/utathya/electricity-consumption).

Just for kicks, we can take this opportunity to compare loading times on CPU with `pandas` and GPU with `cuDF`

```python
path = 'data/electricity_consumption.csv'
```

### On CPU

```python
%%time
# pandas reading time
df = pd.read_csv(path)
```

```python
%%time
from sklearn.preprocessing import MinMaxScaler

#pandas Transform
minmax = MinMaxScaler().fit(df.iloc[:, 7].values.reshape((-1,1)).astype('float32'))
df_time_series = minmax.transform(df.iloc[:, 7].values.reshape((-1,1)).astype('float32')).reshape((-1))
df_time_series = pd.DataFrame(df_time_series)
df_time_series.head()
```

```python
time_original = pd.to_datetime(df.iloc[:, 0]).tolist()
```

### On GPU

```python
%%time
#cudf reading
cudf_data = gd.read_csv(path)
```

```python
%%time
#cudf transform
minmax = MinMaxScaler().fit(cudf_data.iloc[:, 7].to_array().reshape((-1,1)).astype('float32'))
cudf_time_series = minmax.transform(cudf_data.iloc[:, 7].to_array().reshape((-1,1)).astype('float32')).reshape((-1))
cudf_time_series = gd.from_pandas(pd.DataFrame(cudf_time_series))
cudf_time_series.head()
```

```python

```

### Pytorch Encoder

Now what we want to do is to create a representation of data  or an encoding of data (for ex: a intermediate layer in resnet) . So, we will use a simple MLP to do that. 


```python
## Building a Pytorch MLP model to get an intermediate representation of Data

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dimension= 32):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.first_layer_encoder = nn.Linear(input_size, hidden_size)
        self.second_layer_encoder = nn.Linear(hidden_size, dimension)
        self.first_layer_decoder = nn.Linear(dimension, hidden_size)
        self.second_layer_decoder = nn.Linear(hidden_size, input_size)

    def forward(self, input):
        output = nn.functional.relu(self.first_layer_encoder(input))
        output = nn.functional.relu(self.second_layer_encoder(output))
        decode = nn.functional.relu(self.first_layer_decoder(output))
        decode = nn.functional.sigmoid(self.second_layer_decoder(decode))
        return decode, output
```

```python
num_epochs = 100
batch_size = 32
input_size = 1
learning_rate = 0.01

model = Encoder(input_size, hidden_size=32, dimension=32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
```

```python
%%time
from torch.autograd import Variable

# pandas run time
X = Variable(torch.from_numpy(df_time_series.values).float(), requires_grad=False)
for epoch in range(num_epochs):
    output, _ = model(X)
    loss = criterion(output, X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))

```

```python
model = Encoder(input_size, hidden_size=32, dimension=32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
```

```python
%%time
from torch.autograd import Variable

#cudf run time
X = Variable(torch.from_numpy(cudf_time_series.as_matrix()).float(), requires_grad=False)
for epoch in range(num_epochs):
    output, _ = model(X)
    loss = criterion(output, X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data))
```

```python

```

```python
out, encoding = model(X)
encoding.shape
torch.save(encoding, 'electricity_encoding.pt')
```

```python
encoding = torch.load('electricity_encoding.pt')
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
