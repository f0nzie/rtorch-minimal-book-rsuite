
# (PART) Basic Tensor Operations {-}

# Tensors

We describe the most important PyTorch methods in this chapter.



```r
library(rTorch)
```

## Tensor data types


```r
# Default data type
torch$tensor(list(1.2, 3))$dtype  # default for floating point is torch.float32
#> torch.float32
```


```r
# change default data type to float64
torch$set_default_dtype(torch$float64)
torch$tensor(list(1.2, 3))$dtype         # a new floating point tensor
#> torch.float64
```


There are five major type of Tensors in PyTorch


```r
library(rTorch)

byte    <- torch$ByteTensor(3L, 3L)
float   <- torch$FloatTensor(3L, 3L)
double  <- torch$DoubleTensor(3L, 3L)
long    <- torch$LongTensor(3L, 3L)
boolean <- torch$BoolTensor(5L, 5L)

message("byte tensor")
#> byte tensor
byte
#> tensor([[0, 0, 0],
#>         [0, 0, 0],
#>         [0, 0, 0]], dtype=torch.uint8)

message("float tensor")
#> float tensor
float
#> tensor([[0., 0., 0.],
#>         [0., 0., 0.],
#>         [0., 0., 0.]], dtype=torch.float32)

message("double")
#> double
double
#> tensor([[0., 0., 0.],
#>         [0., 0., 0.],
#>         [0., 0., 0.]])

message("long")
#> long
long
#> tensor([[             0,              0,              0],
#>         [             0,              0, 93879514216512],
#>         [             0, 93879511550288,              0]])

message("boolean")
#> boolean
boolean
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [ True, False, False, False,  True]], dtype=torch.bool)
```

A 4D tensor like in MNIST hand-written digits recognition dataset:


```r
mnist_4d <- torch$FloatTensor(60000L, 3L, 28L, 28L)

message("size")
#> size
mnist_4d$size()
#> torch.Size([60000, 3, 28, 28])

message("length")
#> length
length(mnist_4d)
#> [1] 141120000


message("shape, like in numpy")
#> shape, like in numpy
mnist_4d$shape
#> torch.Size([60000, 3, 28, 28])

message("number of elements")
#> number of elements
mnist_4d$numel()
#> [1] 141120000
```

A 3D tensor:


```r
ft3d <- torch$FloatTensor(4L, 3L, 2L)
ft3d
#> tensor([[[1.3329e-34, 3.0630e-41],
#>          [1.4013e-45, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]],
#> 
#>         [[0.0000e+00, 0.0000e+00],
#>          [       nan, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]],
#> 
#>         [[0.0000e+00, 4.4842e-44],
#>          [4.4842e-44, 0.0000e+00],
#>          [1.3329e-34, 3.0630e-41]],
#> 
#>         [[1.4013e-45, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00],
#>          [0.0000e+00, 0.0000e+00]]], dtype=torch.float32)
```


## Arithmetic of tensors

### Add tensors


```r
# add a scalar to a tensor
# 3x5 matrix uniformly distributed between 0 and 1
mat0 <- torch$FloatTensor(3L, 5L)$uniform_(0L, 1L)
mat0 + 0.1
#> tensor([[0.2936, 0.9834, 0.3201, 0.2540, 0.3226],
#>         [0.1328, 0.1580, 0.5292, 0.9398, 0.1906],
#>         [0.2077, 0.2462, 1.0835, 1.0563, 0.6519]], dtype=torch.float32)
```

> The expression ``tensor.index(m)`` is equivalent to ``tensor[m]``.

Add an element of tensor to a tensor:


```r
# fill a 3x5 matrix with 0.1
mat1 <- torch$FloatTensor(3L, 5L)$uniform_(0.1, 0.1)
# a vector with all ones
mat2 <- torch$FloatTensor(5L)$uniform_(1, 1)
mat1[1, 1] + mat2
#> tensor([1.1000, 1.1000, 1.1000, 1.1000, 1.1000], dtype=torch.float32)
```


```r
# add two tensors
mat1 + mat0
#> tensor([[0.2936, 0.9834, 0.3201, 0.2540, 0.3226],
#>         [0.1328, 0.1580, 0.5292, 0.9398, 0.1906],
#>         [0.2077, 0.2462, 1.0835, 1.0563, 0.6519]], dtype=torch.float32)
```

Add two tensors using the function `add()`:

```r
# PyTorch add two tensors
x = torch$rand(5L, 4L)
y = torch$rand(5L, 4L)

print(x$add(y))
#> tensor([[0.3188, 1.0436, 1.1463, 0.7176],
#>         [0.9737, 0.8448, 1.4864, 1.1779],
#>         [1.1231, 0.6356, 0.3719, 0.9354],
#>         [1.4009, 0.4334, 1.4538, 1.7402],
#>         [0.9445, 0.6089, 0.6256, 1.1130]])
```

Add two tensors using the generic `+`:

```r
print(x + y)
#> tensor([[0.3188, 1.0436, 1.1463, 0.7176],
#>         [0.9737, 0.8448, 1.4864, 1.1779],
#>         [1.1231, 0.6356, 0.3719, 0.9354],
#>         [1.4009, 0.4334, 1.4538, 1.7402],
#>         [0.9445, 0.6089, 0.6256, 1.1130]])
```


### Multiply a tensor by a scalar


```r
# Multiply tensor by scalar
tensor = torch$ones(4L, dtype=torch$float64)
scalar = np$float64(4.321)
print(scalar)
#> [1] 4.32
print(torch$scalar_tensor(scalar))
#> tensor(4.3210)
```

Multiply two tensors using the function `mul`:

```r
(prod = torch$mul(tensor, torch$scalar_tensor(scalar)))
#> tensor([4.3210, 4.3210, 4.3210, 4.3210])
```


Short version using generics

```r
(prod = tensor * scalar)
#> tensor([4.3210, 4.3210, 4.3210, 4.3210])
```



## NumPy and PyTorch
`numpy` has been made available as a module in `rTorch`. We can call functions from `numpy` refrerring to it as `np$_a_function`. Examples:


```r
# a 2D numpy array  
syn0 <- np$random$rand(3L, 5L)
print(syn0)
#>       [,1]  [,2]  [,3]   [,4]  [,5]
#> [1,] 0.728 0.869 0.867 0.4852 0.801
#> [2,] 0.660 0.222 0.351 0.0581 0.996
#> [3,] 0.298 0.187 0.113 0.0162 0.896
```



```r
# numpy arrays of zeros
syn1 <- np$zeros(c(5L, 10L))
print(syn1)
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]    0    0    0    0    0    0    0    0    0     0
#> [2,]    0    0    0    0    0    0    0    0    0     0
#> [3,]    0    0    0    0    0    0    0    0    0     0
#> [4,]    0    0    0    0    0    0    0    0    0     0
#> [5,]    0    0    0    0    0    0    0    0    0     0
```


```r
# add a scalar to a numpy array
syn1 = syn1 + 0.1
print(syn1)
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
#> [2,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
#> [3,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
#> [4,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
#> [5,]  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1   0.1
```

### Tuples (Python) and vectors (R)
In numpy a multidimensional array needs to be defined with a tuple
in R we do it with a vector.

In Python, we use a tuple, `(5, 5)`


```python
import numpy as np

print(np.ones((5, 5)))
#> [[1. 1. 1. 1. 1.]
#>  [1. 1. 1. 1. 1.]
#>  [1. 1. 1. 1. 1.]
#>  [1. 1. 1. 1. 1.]
#>  [1. 1. 1. 1. 1.]]
```

In R, we use a vector `c(5L, 5L)`. The `L` indicates an integer.


```r
l1 <- np$ones(c(5L, 5L))
print(l1)
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    1    1    1    1
#> [2,]    1    1    1    1    1
#> [3,]    1    1    1    1    1
#> [4,]    1    1    1    1    1
#> [5,]    1    1    1    1    1
```

Vector-matrix multiplication in numpy:

```r
np$dot(syn0, syn1)
#>       [,1]  [,2]  [,3]  [,4]  [,5]  [,6]  [,7]  [,8]  [,9] [,10]
#> [1,] 0.375 0.375 0.375 0.375 0.375 0.375 0.375 0.375 0.375 0.375
#> [2,] 0.229 0.229 0.229 0.229 0.229 0.229 0.229 0.229 0.229 0.229
#> [3,] 0.151 0.151 0.151 0.151 0.151 0.151 0.151 0.151 0.151 0.151
```

Build a numpy array from three R vectors:

```r
X <- np$array(rbind(c(1,2,3), c(4,5,6), c(7,8,9)))
print(X)
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    4    5    6
#> [3,]    7    8    9
```

And transpose the array:

```r
np$transpose(X)
#>      [,1] [,2] [,3]
#> [1,]    1    4    7
#> [2,]    2    5    8
#> [3,]    3    6    9
```

### Make a numpy array a tensor with `as_tensor()`


```r
a = np$array(list(1, 2, 3))   # a numpy array
t = torch$as_tensor(a)        # convert it to tensor
print(t)
#> tensor([1., 2., 3.])
```

We can create the tensor directly from R using `tensor()`:

```r
torch$tensor(list( 1,  2,  3))   # create a tensor
#> tensor([1., 2., 3.])
t[1L]$fill_(-1)                  # fill element with -1
#> tensor(-1.)
print(a)
#> [1] -1  2  3
```


### Tensor to array, and viceversa
This is a very common operation in machine learning:


```r
# convert tensor to a numpy array
a = torch$rand(5L, 4L)
b = a$numpy()
print(b)
#>       [,1]  [,2]   [,3]  [,4]
#> [1,] 0.703 0.941 0.5242 0.431
#> [2,] 0.662 0.855 0.6995 0.771
#> [3,] 0.289 0.147 0.6468 0.479
#> [4,] 0.810 0.411 0.1984 0.514
#> [5,] 0.289 0.036 0.0623 0.802
```


```r
# convert a numpy array to a tensor
np_a = np$array(c(c(3, 4), c(3, 6)))
t_a = torch$from_numpy(np_a)
print(t_a)
#> tensor([3., 4., 3., 6.])
```


## Create tensors

A random 1D tensor:

```r
ft1 <- torch$FloatTensor(np$random$rand(5L))
print(ft1)
#> tensor([0.4131, 0.0010, 0.5875, 0.6724, 0.3338], dtype=torch.float32)
```

Force a tensor as a float of 64-bits:

```r
ft2 <- torch$as_tensor(np$random$rand(5L), dtype= torch$float64)
print(ft2)
#> tensor([0.2407, 0.6178, 0.5876, 0.0568, 0.9463])
```

Convert the tensor to float 16-bits:

```r
ft2_dbl <- torch$as_tensor(ft2, dtype = torch$float16)
ft2_dbl
#> tensor([0.2407, 0.6177, 0.5874, 0.0568, 0.9463], dtype=torch.float16)
```


Create a tensor of size (5 x 7) with uninitialized memory:


```r
a <- torch$FloatTensor(5L, 7L)
print(a)
#> tensor([[0., 0., 0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0., 0., 0.]], dtype=torch.float32)
```

Using arange to create a tensor. Start from 0:

```r
v = torch$arange(9L)
(v = v$view(3L, 3L))
#> tensor([[0, 1, 2],
#>         [3, 4, 5],
#>         [6, 7, 8]])
```



## Tensor resizing


```r
x = torch$randn(2L, 3L)            # Size 2x3
y = x$view(6L)                     # Resize x to size 6
z = x$view(-1L, 2L)                # Size 3x2
print(y)
#> tensor([-0.9516,  1.0804,  0.1661,  0.3614, -1.3280, -0.1116])
print(z)
#> tensor([[-0.9516,  1.0804],
#>         [ 0.1661,  0.3614],
#>         [-1.3280, -0.1116]])
```

Reproduce this tensor:

```
 0 1 2
 3 4 5
 6 7 8
``` 


```r
v = torch$arange(9L)
(v = v$view(3L, 3L))
#> tensor([[0, 1, 2],
#>         [3, 4, 5],
#>         [6, 7, 8]])
```

### Concatenate tensors


```r
x = torch$randn(2L, 3L)
print(x)
#> tensor([[ 1.1317, -0.1802,  0.9361],
#>         [ 1.7742,  0.1528,  1.2348]])
```

Concatenate tensors by `dim=0`:

```r
torch$cat(list(x, x, x), 0L)
#> tensor([[ 1.1317, -0.1802,  0.9361],
#>         [ 1.7742,  0.1528,  1.2348],
#>         [ 1.1317, -0.1802,  0.9361],
#>         [ 1.7742,  0.1528,  1.2348],
#>         [ 1.1317, -0.1802,  0.9361],
#>         [ 1.7742,  0.1528,  1.2348]])
```


Concatenate tensors by `dim=1`:

```r
torch$cat(list(x, x, x), 1L)
#> tensor([[ 1.1317, -0.1802,  0.9361,  1.1317, -0.1802,  0.9361,  1.1317, -0.1802,
#>           0.9361],
#>         [ 1.7742,  0.1528,  1.2348,  1.7742,  0.1528,  1.2348,  1.7742,  0.1528,
#>           1.2348]])
```


## Reshape tensors

### With function `chunk()`:

Let's say this is an image tensor with the 3-channels and 28x28 pixels


```r
# ----- Reshape tensors -----
img <- torch$ones(3L, 28L, 28L)  # Create the tensor of ones
print(img$size())
#> torch.Size([3, 28, 28])
```

On the first dimension `dim = 0L`, reshape the tensor:

```r
img_chunks <- torch$chunk(img, chunks = 3L, dim = 0L)
print(length(img_chunks))
#> [1] 3
```

The first chunk member:

```r
# 1st chunk member
img_chunk <- img_chunks[[1]]
print(img_chunk$size())
#> torch.Size([1, 28, 28])
print(img_chunk$sum())      # if the tensor had all ones, what is the sum?
#> tensor(784.)
```

The second chunk member:

```r
# 2nd chunk member
img_chunk <- img_chunks[[2]]
print(img_chunk$size())
#> torch.Size([1, 28, 28])
print(img_chunk$sum())        # if the tensor had all ones, what is the sum?
#> tensor(784.)
```


```r
# 3rd chunk member
img_chunk <- img_chunks[[3]]
print(img_chunk$size())
#> torch.Size([1, 28, 28])
print(img_chunk$sum())        # if the tensor had all ones, what is the sum?
#> tensor(784.)
```


### With `index_select()`:


```r
img <- torch$ones(3L, 28L, 28L)  # Create the tensor of ones
```


This is the layer 1:

```r
# index_select. get layer 1
indices = torch$tensor(c(0L))
img_layer <- torch$index_select(img, dim = 0L, index = indices)
```

The size of the layer:

```r
print(img_layer$size())
#> torch.Size([1, 28, 28])
```

The sum of all elements in that layer:

```r
print(img_layer$sum())
#> tensor(784.)
```


This is the layer 2:

```r
# index_select. get layer 2
indices = torch$tensor(c(1L))
img_layer <- torch$index_select(img, dim = 0L, index = indices)
print(img_layer$size())
#> torch.Size([1, 28, 28])
print(img_layer$sum())
#> tensor(784.)
```


This is the layer 3:

```r
# index_select. get layer 3
indices = torch$tensor(c(2L))
img_layer <- torch$index_select(img, dim = 0L, index = indices)
print(img_layer$size())
#> torch.Size([1, 28, 28])
print(img_layer$sum())
#> tensor(784.)
```



## Special tensors

### Identity matrix


```r
# identity matrix
eye = torch$eye(3L)              # Create an identity 3x3 tensor
print(eye)
#> tensor([[1., 0., 0.],
#>         [0., 1., 0.],
#>         [0., 0., 1.]])
```

### Ones


```r
(v = torch$ones(10L))              # A tensor of size 10 containing all ones
#> tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
(v = torch$ones(2L, 1L, 2L, 1L))      # Size 2x1x2x1
#> tensor([[[[1.],
#>           [1.]]],
#> 
#> 
#>         [[[1.],
#>           [1.]]]])
```


```r
v = torch$ones_like(eye)     # A tensor with same shape as eye. Fill it with 1.
v
#> tensor([[1., 1., 1.],
#>         [1., 1., 1.],
#>         [1., 1., 1.]])
```

### Zeros


```r
(z = torch$zeros(10L))             # A tensor of size 10 containing all zeros
#> tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```


## Tensor fill

On this tensor:

```r
(v = torch$ones(3L, 3L))
#> tensor([[1., 1., 1.],
#>         [1., 1., 1.],
#>         [1., 1., 1.]])
```

Fill row 1 with 2s:

```r
v[1L, ]$fill_(2L)         
#> tensor([2., 2., 2.])
print(v)
#> tensor([[2., 2., 2.],
#>         [1., 1., 1.],
#>         [1., 1., 1.]])
```

Fill row 2 with 3s:

```r
v[2L, ]$fill_(3L)       
#> tensor([3., 3., 3.])
print(v)
#> tensor([[2., 2., 2.],
#>         [3., 3., 3.],
#>         [1., 1., 1.]])
```


```r
# Initialize Tensor with a range of value
v = torch$arange(10L)             # similar to range(5) but creating a Tensor
(v = torch$arange(0L, 10L, step = 1L))  # Size 5. Similar to range(0, 5, 1)
#> tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```


### Initialize a linear or log scale Tensor

Create a tensor with 10 linear points for (1, 10) inclusive:

```r
(v = torch$linspace(1L, 10L, steps = 10L)) 
#> tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
```

Create a tensor with 10 logarithmic points for (1, 10) inclusive:

```r
(v = torch$logspace(start=-10L, end = 10L, steps = 5L)) 
#> tensor([1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10])
```


### Inplace / Out-of-place

On this uninitialized tensor:

```r
(a <- torch$FloatTensor(5L, 7L))
#> tensor([[ 1.5259e-16,  4.5715e-41, -1.7633e+38,  3.0628e-41,  1.4013e-45,
#>           0.0000e+00,  2.9867e-36],
#>         [ 3.0630e-41,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  6.3058e-44,  6.3058e-44,  1.5465e-34,
#>           3.0630e-41,  1.4013e-45],
#>         [ 0.0000e+00,  1.2463e-34,  3.0630e-41,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00],
#>         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#>           0.0000e+00,  0.0000e+00]], dtype=torch.float32)
```

Fill the tensor with the value 3.5:

```r
a$fill_(3.5)
#> tensor([[3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000]],
#>        dtype=torch.float32)
```

Add a scalar to the tensor:

```r
b <- a$add(4.0)
```

The tensor `a` is still filled with 3.5.
A new tensor `b` is returned with values 3.5 + 4.0 = 7.5


```r
print(a)
#> tensor([[3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000],
#>         [3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000, 3.5000]],
#>        dtype=torch.float32)
print(b)
#> tensor([[7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000],
#>         [7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000, 7.5000]],
#>        dtype=torch.float32)
```



## Access to tensor elements


```r
# replace an element at position 0, 0
(new_tensor = torch$Tensor(list(list(1, 2), list(3, 4))))
#> tensor([[1., 2.],
#>         [3., 4.]])
```

Print element at position `1,1`:

```r
print(new_tensor[1L, 1L])
#> tensor(1.)
```

Fill element at position `1,1` with 5:

```r
new_tensor[1L, 1L]$fill_(5)
#> tensor(5.)
```

Show the modified tensor:

```r
print(new_tensor)   # tensor([[ 5.,  2.],[ 3.,  4.]])
#> tensor([[5., 2.],
#>         [3., 4.]])
```


Access an element at position `1, 0`:

```r
print(new_tensor[2L, 1L])           # tensor([ 3.])
#> tensor(3.)
print(new_tensor[2L, 1L]$item())    # 3.
#> [1] 3
```


### Using indices to access elements

On this tensor:

```r
x = torch$randn(3L, 4L)
print(x)
#> tensor([[ 0.2679,  0.8287,  1.3341, -0.2913],
#>         [-0.8384,  1.4108, -0.9045, -0.9782],
#>         [-0.4219,  1.9641, -1.3441, -0.2731]])
```

Select indices, `dim=0`:

```r
indices = torch$tensor(list(0L, 2L))
torch$index_select(x, 0L, indices)
#> tensor([[ 0.2679,  0.8287,  1.3341, -0.2913],
#>         [-0.4219,  1.9641, -1.3441, -0.2731]])
```

Select indices, `dim=1`:

```r
torch$index_select(x, 1L, indices)
#> tensor([[ 0.2679,  1.3341],
#>         [-0.8384, -0.9045],
#>         [-0.4219, -1.3441]])
```


### Using the `take` function


```r
# Take by indices
src = torch$tensor(list(list(4, 3, 5),
                        list(6, 7, 8)) )
print(src)
#> tensor([[4., 3., 5.],
#>         [6., 7., 8.]])
print( torch$take(src, torch$tensor(list(0L, 2L, 5L))) )
#> tensor([4., 5., 8.])
```


## Other tensor operations

### Cross product

```r
m1 = torch$ones(3L, 5L)
m2 = torch$ones(3L, 5L)
v1 = torch$ones(3L)
# Cross product
# Size 3x5
(r = torch$cross(m1, m2))
#> tensor([[0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0.],
#>         [0., 0., 0., 0., 0.]])
```

### Dot product


```r
# Dot product of 2 tensors
# Dot product of 2 tensors

p <- torch$Tensor(list(4L, 2L))
q <- torch$Tensor(list(3L, 1L))                   

(r = torch$dot(p, q)) # 14
#> tensor(14.)
(r <- p %.*% q)
#> tensor(14.)
```


## Logical operations


```r
m0 = torch$zeros(3L, 5L)
m1 = torch$ones(3L, 5L)
m2 = torch$eye(3L, 5L)

print(m1 == m0)
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]], dtype=torch.bool)
```


```r
print(m1 != m1)
#> tensor([[False, False, False, False, False],
#>         [False, False, False, False, False],
#>         [False, False, False, False, False]], dtype=torch.bool)
```


```r
print(m2 == m2)
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]], dtype=torch.bool)
```



```r
# AND
m1 & m1
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]], dtype=torch.bool)
```


```r
# OR
m0 | m2
#> tensor([[ True, False, False, False, False],
#>         [False,  True, False, False, False],
#>         [False, False,  True, False, False]], dtype=torch.bool)
```


```r
# OR
m1 | m2
#> tensor([[True, True, True, True, True],
#>         [True, True, True, True, True],
#>         [True, True, True, True, True]], dtype=torch.bool)
```


```r
# all_boolean <- function(x) {
#   # convert tensor of 1s and 0s to a unique boolean
#   as.logical(torch$all(x)$numpy())
# }

# tensor is less than
A <- torch$ones(60000L, 1L, 28L, 28L)
C <- A * 0.5

# is C < A
all(torch$lt(C, A))
#> tensor(1, dtype=torch.uint8)
all(C < A)
#> tensor(1, dtype=torch.uint8)
# is A < C
all(A < C)
#> tensor(0, dtype=torch.uint8)
```


```r
# tensor is greater than
A <- torch$ones(60000L, 1L, 28L, 28L)
D <- A * 2.0
all(torch$gt(D, A))
#> tensor(1, dtype=torch.uint8)
all(torch$gt(A, D))
#> tensor(0, dtype=torch.uint8)
```



```r
# tensor is less than or equal
A1 <- torch$ones(60000L, 1L, 28L, 28L)
all(torch$le(A1, A1))
#> tensor(1, dtype=torch.uint8)
all(A1 <= A1)
#> tensor(1, dtype=torch.uint8)

# tensor is greater than or equal
A0 <- torch$zeros(60000L, 1L, 28L, 28L)
all(torch$ge(A0, A0))
#> tensor(1, dtype=torch.uint8)
all(A0 >= A0)
#> tensor(1, dtype=torch.uint8)

all(A1 >= A0)
#> tensor(1, dtype=torch.uint8)
all(A1 <= A0)
#> tensor(0, dtype=torch.uint8)
```

### Logical NOT


```r
all_true <- torch$BoolTensor(list(TRUE, TRUE, TRUE, TRUE))
all_true
#> tensor([True, True, True, True], dtype=torch.bool)

# logical NOT
not_all_true <- !all_true
not_all_true
#> tensor([False, False, False, False], dtype=torch.bool)
```


```r
diag <- torch$eye(5L)
diag
#> tensor([[1., 0., 0., 0., 0.],
#>         [0., 1., 0., 0., 0.],
#>         [0., 0., 1., 0., 0.],
#>         [0., 0., 0., 1., 0.],
#>         [0., 0., 0., 0., 1.]])

# logical NOT
not_diag <- !diag

# convert to integer
not_diag$to(dtype=torch$uint8)
#> tensor([[0, 1, 1, 1, 1],
#>         [1, 0, 1, 1, 1],
#>         [1, 1, 0, 1, 1],
#>         [1, 1, 1, 0, 1],
#>         [1, 1, 1, 1, 0]], dtype=torch.uint8)
```


## Distributions

Initialize a tensor randomized with a normal distribution with mean=0, var=1:


```r
a  <- torch$randn(5L, 7L)
print(a)
#> tensor([[ 1.3544, -1.8241, -0.7558, -0.1505, -1.8054, -0.1313, -0.7245],
#>         [ 0.6932, -0.5829,  2.4714,  1.2994, -1.2779, -0.0682, -0.6886],
#>         [-0.3000,  0.3478, -1.1571,  0.6252, -0.7140, -0.9329, -0.9316],
#>         [-1.6713,  0.3191,  0.7264, -1.6243,  1.0500,  0.2943,  0.5875],
#>         [-0.4391,  0.5498, -0.1649, -2.5375, -1.6491, -0.8779,  0.2126]])
print(a$size())
#> torch.Size([5, 7])
```

### Uniform matrix


```r
library(rTorch)

# 3x5 matrix uniformly distributed between 0 and 1
mat0 <- torch$FloatTensor(3L, 5L)$uniform_(0L, 1L)

# fill a 3x5 matrix with 0.1
mat1 <- torch$FloatTensor(3L, 5L)$uniform_(0.1, 0.1)

# a vector with all ones
mat2 <- torch$FloatTensor(5L)$uniform_(1, 1)

mat0
#> tensor([[0.2317, 0.4236, 0.3170, 0.8195, 0.3516],
#>         [0.7720, 0.8723, 0.0118, 0.8298, 0.4079],
#>         [0.8013, 0.6037, 0.2871, 0.4604, 0.0931]], dtype=torch.float32)
mat1
#> tensor([[0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#>         [0.1000, 0.1000, 0.1000, 0.1000, 0.1000]], dtype=torch.float32)
```

### Binomial distribution


```r
Binomial <- torch$distributions$binomial$Binomial

m = Binomial(100, torch$tensor(list(0 , .2, .8, 1)))
(x = m$sample())
#> tensor([  0.,  21.,  82., 100.])
```


```r
m = Binomial(torch$tensor(list(list(5.), list(10.))), 
             torch$tensor(list(0.5, 0.8)))
(x = m$sample())
#> tensor([[1., 4.],
#>         [3., 9.]])
```

### Exponential distribution


```r
Exponential <- torch$distributions$exponential$Exponential

m = Exponential(torch$tensor(list(1.0)))
m$sample()  # Exponential distributed with rate=1
#> tensor([1.5468])
```

### Weibull distribution


```r
Weibull <- torch$distributions$weibull$Weibull

m = Weibull(torch$tensor(list(1.0)), torch$tensor(list(1.0)))
m$sample()  # sample from a Weibull distribution with scale=1, concentration=1
#> tensor([0.3572])
```
