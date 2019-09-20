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

## RAPIDS.ai and Deep Learning with PyTorch

In this notebook, we'll tackle the compatibility of the [RAPIDS.ai]() tools with major Deep Learning framework, i.e. PyTorch.

We'll use an example dataset of relatively small size, just to show the improvements you can achieve already. 


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
```

### On GPU

```python
%%time
#cudf reading
cudf_data = gd.read_csv(path)
```

```python
%%time
from sklearn.preprocessing import MinMaxScaler

#cudf transform
minmax = MinMaxScaler().fit(cudf_data.iloc[:, 7].to_array().reshape((-1,1)).astype('float32'))
cudf_time_series = minmax.transform(cudf_data.iloc[:, 7].to_array().reshape((-1,1)).astype('float32')).reshape((-1))
cudf_time_series = gd.from_pandas(pd.DataFrame(cudf_time_series))
```

---
We see here that loading data from CSV is ~4 times faster on GPU. However, doing the `MinMaxScaler` operation is longer (by factor ~4 as well) because it is not available on cuDF : you need to fall back on CPU and pandas, multiplying the memory changes.

### Pytorch Encoder

Now what we want to do is to create a representation of data  or an encoding of data (for ex: a intermediate layer in resnet) . So, we will use a simple MLP autoencoder to do that. 


```python
## Building a Pytorch MLP model to get an intermediate representation of Data

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.dlpack import from_dlpack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
# Build the network

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
        decode = torch.sigmoid(self.second_layer_decoder(decode))
        return decode, output
```

```python
# Define the training loop
num_epochs = 100
batch_size = 32
input_size = 1
learning_rate = 0.01


def train_model(X, device):
    
    model = Encoder(input_size, hidden_size=32, dimension=32).to(device)
    _X = X.to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    
    for epoch in range(num_epochs):
        output, _ = model(_X)
        loss = criterion(output, _X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch) % 10 == 0:
            print('epoch [{:02d}/{}], loss:{:.9f}'.format(epoch, num_epochs, loss.data))

    return model

```

## Loading the DataFrame into a PyTorch Tensor

Now that we have done our preprocessing and defined the training algorithm, we need to give it an input. 

We will run the training on GPU, so we need to create a GPU Tensor.

If we have a pandas DataFrame we can load it with `torch.from_numpy()` and send it to the GPU with `X.to("cuda:0")`. Easy!

Now, pytorch does not have a `from_cudf()` or `from_cupy()` method. If we want to combine PyTorch with a cuDF DataFrame, we could be naive and go back to a CPU numpy array, then convert this to a Tensor.

As we can guess, and what we'll see below, this is far from optimal. So how can we go faster ?

Thankfully, there is a package/format called [DLPack](https://github.com/dmlc/dlpack) that aims to do exactly this: transfer data while keeping it on GPU memory. It's built-in in PyTorch and in cuDF so it provides a nice alternative. The additional conversion step is still there, but we are now going pretty much as fast as the first example.

```python
%%time 
# Create a Tensor from a pandas DataFrame
X_pandas = Variable(torch.from_numpy(df_time_series.values).float(), requires_grad=False).to(device)
```

```python
%%time
# Create a Tensor from a cuDF array 
# The naive way : go back on CPU with .as_matrix() then cast it back to GPU
X_numpy = Variable(torch.from_numpy(cudf_time_series.as_matrix()).float(), requires_grad=False).to(device)
```

```python
%%time
# Create a Tensor from a cuDF array 
# The right way : use the GPU-memory format DLPack handled by both cuDF and PyTorch

X_dlpack = from_dlpack(cudf_time_series.to_dlpack()).unsqueeze(1).to(device)
```

## Comparing training times

Now all the three Tensors `X_pandas`, `X_numpy` and `X_dlpack` are essentially identical, we can do the classical _Training-on-CPU_ vs _Training-on-CPU_ time comparison. We get a more than 2x speedup factor.

```python
%%time
# Train on CPU
model = train_model(X_dlpack, 'cpu')
```

```python
%%time
# Train on CPU
model = train_model(X_dlpack, 'cuda:0')
```

# End-to-end pipeline

After comparing times step-by-step, let's compare a loading-to-training pipeline on CPU and GPU, let's start with CPU

## CPU

```python
%reset -f
```

```python
%%time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.dlpack import from_dlpack

path = 'data/electricity_consumption.csv'

# Load Data
df = pd.read_csv(path)

#Transform
minmax = MinMaxScaler().fit(df.iloc[:, 7].values.reshape((-1,1)).astype('float32'))
df_time_series = minmax.transform(df.iloc[:, 7].values.reshape((-1,1)).astype('float32')).reshape((-1))
df_time_series = pd.DataFrame(df_time_series)

# Build the network
device = "cpu"

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
        decode = torch.sigmoid(self.second_layer_decoder(decode))
        return decode, output
    
# Define the training loop
num_epochs = 100
batch_size = 32
input_size = 1
learning_rate = 0.01


def train_model(X, device):
    
    model = Encoder(input_size, hidden_size=32, dimension=32).to(device)
    _X = X.to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    
    for epoch in range(num_epochs):
        output, _ = model(_X)
        loss = criterion(output, _X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch) % 10 == 0:
            print('epoch [{:02d}/{}], loss:{:.9f}'.format(epoch, num_epochs, loss.data))

    return model

# Load data into Tensor
X = Variable(torch.from_numpy(df_time_series.values).float(), requires_grad=False).to(device)

# Train model
model = train_model(X, device)
```

## GPU
Now time to run the same on GPU

```python
%reset -f
```

```python
%%time
import pandas as pd
import cudf as gd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.dlpack import from_dlpack

path = 'data/electricity_consumption.csv'

cudf_data = gd.read_csv(path)

#cudf transform
minmax = MinMaxScaler().fit(cudf_data.iloc[:, 7].to_array().reshape((-1,1)).astype('float32'))
cudf_time_series = minmax.transform(cudf_data.iloc[:, 7].to_array().reshape((-1,1)).astype('float32')).reshape((-1))
cudf_time_series = gd.from_pandas(pd.DataFrame(cudf_time_series))


# Build the network
device = "cuda:0"

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
        decode = torch.sigmoid(self.second_layer_decoder(decode))
        return decode, output
    
# Define the training loop
num_epochs = 100
batch_size = 32
input_size = 1
learning_rate = 0.01


def train_model(X, device):
    
    model = Encoder(input_size, hidden_size=32, dimension=32).to(device)
    _X = X.to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    
    for epoch in range(num_epochs):
        output, _ = model(_X)
        loss = criterion(output, _X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch) % 10 == 0:
            print('epoch [{:02d}/{}], loss:{:.9f}'.format(epoch, num_epochs, loss.data))

    return model

# Load data into Tensor
X = from_dlpack(cudf_time_series.to_dlpack()).unsqueeze(1).to(device)

# Train model
model = train_model(X, device)
```

## Conclusion
We get an overall 2.5 times faster pipeline on GPU with all packages rather than on CPU. Another reason to consider using the RAPIDS.ai suite !
