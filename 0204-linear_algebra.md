
# Linear Algebra with Torch {#linearalgebra}

The following are basic operations of Linear Algebra using PyTorch.



```r
library(rTorch)
```


## Scalars


```r
torch$scalar_tensor(2.78654)
#> tensor(2.7865)

torch$scalar_tensor(0L)
#> tensor(0.)

torch$scalar_tensor(1L)
#> tensor(1.)

torch$scalar_tensor(TRUE)
#> tensor(1.)

torch$scalar_tensor(FALSE)
#> tensor(0.)
```

## Vectors


```r
v <- c(0, 1, 2, 3, 4, 5)
torch$as_tensor(v)
#> tensor([0., 1., 2., 3., 4., 5.])
```



```r
# row-vector
message("R matrix")
#> R matrix
(mr <- matrix(1:10, nrow=1))
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]    1    2    3    4    5    6    7    8    9    10
message("as_tensor")
#> as_tensor
torch$as_tensor(mr)
#> tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]], dtype=torch.int32)
message("shape_of_tensor")
#> shape_of_tensor
torch$as_tensor(mr)$shape
#> torch.Size([1, 10])
```


```r
# column-vector
message("R matrix, one column")
#> R matrix, one column
(mc <- matrix(1:10, ncol=1))
#>       [,1]
#>  [1,]    1
#>  [2,]    2
#>  [3,]    3
#>  [4,]    4
#>  [5,]    5
#>  [6,]    6
#>  [7,]    7
#>  [8,]    8
#>  [9,]    9
#> [10,]   10
message("as_tensor")
#> as_tensor
torch$as_tensor(mc)
#> tensor([[ 1],
#>         [ 2],
#>         [ 3],
#>         [ 4],
#>         [ 5],
#>         [ 6],
#>         [ 7],
#>         [ 8],
#>         [ 9],
#>         [10]], dtype=torch.int32)
message("size of tensor")
#> size of tensor
torch$as_tensor(mc)$shape
#> torch.Size([10, 1])
```

## Matrices


```r
message("R matrix")
#> R matrix
(m1 <- matrix(1:24, nrow = 3, byrow = TRUE))
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
#> [1,]    1    2    3    4    5    6    7    8
#> [2,]    9   10   11   12   13   14   15   16
#> [3,]   17   18   19   20   21   22   23   24
message("as_tensor")
#> as_tensor
(t1 <- torch$as_tensor(m1))
#> tensor([[ 1,  2,  3,  4,  5,  6,  7,  8],
#>         [ 9, 10, 11, 12, 13, 14, 15, 16],
#>         [17, 18, 19, 20, 21, 22, 23, 24]], dtype=torch.int32)
message("shape")
#> shape
torch$as_tensor(m1)$shape
#> torch.Size([3, 8])
message("size")
#> size
torch$as_tensor(m1)$size()
#> torch.Size([3, 8])
message("dim")
#> dim
dim(torch$as_tensor(m1))
#> [1] 3 8
message("length")
#> length
length(torch$as_tensor(m1))
#> [1] 24
```


```r
message("R matrix")
#> R matrix
(m2 <- matrix(0:99, ncol = 10))
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0   10   20   30   40   50   60   70   80    90
#>  [2,]    1   11   21   31   41   51   61   71   81    91
#>  [3,]    2   12   22   32   42   52   62   72   82    92
#>  [4,]    3   13   23   33   43   53   63   73   83    93
#>  [5,]    4   14   24   34   44   54   64   74   84    94
#>  [6,]    5   15   25   35   45   55   65   75   85    95
#>  [7,]    6   16   26   36   46   56   66   76   86    96
#>  [8,]    7   17   27   37   47   57   67   77   87    97
#>  [9,]    8   18   28   38   48   58   68   78   88    98
#> [10,]    9   19   29   39   49   59   69   79   89    99
message("as_tensor")
#> as_tensor
(t2 <- torch$as_tensor(m2))
#> tensor([[ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#>         [ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
#>         [ 2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
#>         [ 3, 13, 23, 33, 43, 53, 63, 73, 83, 93],
#>         [ 4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
#>         [ 5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
#>         [ 6, 16, 26, 36, 46, 56, 66, 76, 86, 96],
#>         [ 7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
#>         [ 8, 18, 28, 38, 48, 58, 68, 78, 88, 98],
#>         [ 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]], dtype=torch.int32)
message("shape")
#> shape
t2$shape
#> torch.Size([10, 10])
message("dim")
#> dim
dim(torch$as_tensor(m2))
#> [1] 10 10
```


```r
m1[1, 1]
#> [1] 1
m2[1, 1]
#> [1] 0
```


```r
t1[1, 1]
#> tensor(1, dtype=torch.int32)
t2[1, 1]
#> tensor(0, dtype=torch.int32)
```

## 3D+ tensors


```r
# RGB color image has three axes 
(img <- torch$rand(3L, 28L, 28L))
#> tensor([[[0.6838, 0.3793, 0.3066,  ..., 0.0474, 0.5715, 0.9359],
#>          [0.4573, 0.2914, 0.9174,  ..., 0.6256, 0.0516, 0.9690],
#>          [0.9133, 0.7387, 0.3450,  ..., 0.5241, 0.1441, 0.4213],
#>          ...,
#>          [0.5118, 0.3353, 0.3753,  ..., 0.3680, 0.0961, 0.0167],
#>          [0.5611, 0.9762, 0.4540,  ..., 0.9749, 0.0483, 0.3361],
#>          [0.1310, 0.1717, 0.7753,  ..., 0.1157, 0.2699, 0.1538]],
#> 
#>         [[0.4654, 0.0150, 0.4014,  ..., 0.5985, 0.1321, 0.7614],
#>          [0.2011, 0.8940, 0.9488,  ..., 0.8859, 0.2629, 0.0767],
#>          [0.1636, 0.4984, 0.3436,  ..., 0.3952, 0.9503, 0.0591],
#>          ...,
#>          [0.9214, 0.8735, 0.7682,  ..., 0.0400, 0.8388, 0.0039],
#>          [0.0269, 0.4981, 0.3315,  ..., 0.1393, 0.4570, 0.4018],
#>          [0.5696, 0.1619, 0.5168,  ..., 0.2311, 0.9235, 0.5666]],
#> 
#>         [[0.1667, 0.6523, 0.2396,  ..., 0.2067, 0.6236, 0.3981],
#>          [0.8434, 0.9325, 0.8721,  ..., 0.6559, 0.0022, 0.6942],
#>          [0.5514, 0.8077, 0.5026,  ..., 0.5987, 0.4867, 0.7835],
#>          ...,
#>          [0.2976, 0.0737, 0.7688,  ..., 0.8451, 0.0504, 0.8713],
#>          [0.9641, 0.7900, 0.8982,  ..., 0.6912, 0.2741, 0.8682],
#>          [0.4518, 0.9062, 0.5950,  ..., 0.8578, 0.6434, 0.9677]]])
img$shape
#> torch.Size([3, 28, 28])
```


```r
img[1, 1, 1]
#> tensor(0.6838)
img[3, 28, 28]
#> tensor(0.9677)
```


## Transpose of a matrix


```r
(m3 <- matrix(1:25, ncol = 5))
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    6   11   16   21
#> [2,]    2    7   12   17   22
#> [3,]    3    8   13   18   23
#> [4,]    4    9   14   19   24
#> [5,]    5   10   15   20   25

# transpose
message("transpose")
#> transpose
tm3 <- t(m3)
tm3
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    2    3    4    5
#> [2,]    6    7    8    9   10
#> [3,]   11   12   13   14   15
#> [4,]   16   17   18   19   20
#> [5,]   21   22   23   24   25
```


```r
message("as_tensor")
#> as_tensor
(t3 <- torch$as_tensor(m3))
#> tensor([[ 1,  6, 11, 16, 21],
#>         [ 2,  7, 12, 17, 22],
#>         [ 3,  8, 13, 18, 23],
#>         [ 4,  9, 14, 19, 24],
#>         [ 5, 10, 15, 20, 25]], dtype=torch.int32)
message("transpose")
#> transpose
tt3 <- t3$transpose(dim0 = 0L, dim1 = 1L)
tt3
#> tensor([[ 1,  2,  3,  4,  5],
#>         [ 6,  7,  8,  9, 10],
#>         [11, 12, 13, 14, 15],
#>         [16, 17, 18, 19, 20],
#>         [21, 22, 23, 24, 25]], dtype=torch.int32)
```


```r
tm3 == tt3$numpy()   # convert first the tensor to numpy
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,] TRUE TRUE TRUE TRUE TRUE
#> [2,] TRUE TRUE TRUE TRUE TRUE
#> [3,] TRUE TRUE TRUE TRUE TRUE
#> [4,] TRUE TRUE TRUE TRUE TRUE
#> [5,] TRUE TRUE TRUE TRUE TRUE
```

## Vectors, special case of a matrix


```r
message("R matrix")
#> R matrix
m2 <- matrix(0:99, ncol = 10)
message("as_tensor")
#> as_tensor
(t2 <- torch$as_tensor(m2))
#> tensor([[ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#>         [ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
#>         [ 2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
#>         [ 3, 13, 23, 33, 43, 53, 63, 73, 83, 93],
#>         [ 4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
#>         [ 5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
#>         [ 6, 16, 26, 36, 46, 56, 66, 76, 86, 96],
#>         [ 7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
#>         [ 8, 18, 28, 38, 48, 58, 68, 78, 88, 98],
#>         [ 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]], dtype=torch.int32)

# in R
message("select column of matrix")
#> select column of matrix
(v1 <- m2[, 1])
#>  [1] 0 1 2 3 4 5 6 7 8 9
message("select row of matrix")
#> select row of matrix
(v2 <- m2[10, ])
#>  [1]  9 19 29 39 49 59 69 79 89 99
```


```r
# PyTorch
message()
#> 
t2c <- t2[, 1]
t2r <- t2[10, ]

t2c
#> tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
t2r
#> tensor([ 9, 19, 29, 39, 49, 59, 69, 79, 89, 99], dtype=torch.int32)
```

In vectors, the vector and its transpose are equal.


```r
tt2r <- t2r$transpose(dim0 = 0L, dim1 = 0L)
tt2r
#> tensor([ 9, 19, 29, 39, 49, 59, 69, 79, 89, 99], dtype=torch.int32)
```


```r
# a tensor of booleans. is vector equal to its transposed?
t2r == tt2r
#> tensor([True, True, True, True, True, True, True, True, True, True],
#>        dtype=torch.bool)
```

## Tensor arithmetic


```r
message("x")
#> x
(x = torch$ones(5L, 4L))
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
message("y")
#> y
(y = torch$ones(5L, 4L))
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
message("x+y")
#> x+y
x + y
#> tensor([[2., 2., 2., 2.],
#>         [2., 2., 2., 2.],
#>         [2., 2., 2., 2.],
#>         [2., 2., 2., 2.],
#>         [2., 2., 2., 2.]])
```

$$A + B = B + A$$


```r
x + y == y + x
#> tensor([[True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True]], dtype=torch.bool)
```

## Add a scalar to a tensor


```r
s <- 0.5    # scalar
x + s
#> tensor([[1.5000, 1.5000, 1.5000, 1.5000],
#>         [1.5000, 1.5000, 1.5000, 1.5000],
#>         [1.5000, 1.5000, 1.5000, 1.5000],
#>         [1.5000, 1.5000, 1.5000, 1.5000],
#>         [1.5000, 1.5000, 1.5000, 1.5000]])
```


```r
# scalar multiplying two tensors
s * (x + y)
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
```

## Multiplying tensors

$$A * B = B * A$$


```r
message("x")
#> x
(x = torch$ones(5L, 4L))
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
message("y")
#> y
(y = torch$ones(5L, 4L))
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
message("2x+4y")
#> 2x+4y
(z = 2 * x + 4 * y)
#> tensor([[6., 6., 6., 6.],
#>         [6., 6., 6., 6.],
#>         [6., 6., 6., 6.],
#>         [6., 6., 6., 6.],
#>         [6., 6., 6., 6.]])
```



```r
x * y == y * x
#> tensor([[True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True]], dtype=torch.bool)
```



## Dot product

$$dot(a,b)_{i,j,k,a,b,c} = \sum_m a_{i,j,k,m}b_{a,b,m,c}$$


```r
torch$dot(torch$tensor(c(2, 3)), torch$tensor(c(2, 1)))
#> tensor(7.)
```


```r
a <- np$array(list(list(1, 2), list(3, 4)))
a
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    3    4
b <- np$array(list(list(1, 2), list(3, 4)))
b
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    3    4

np$dot(a, b)
#>      [,1] [,2]
#> [1,]    7   10
#> [2,]   15   22
```

`torch.dot()` treats both a and b as 1D vectors (irrespective of their original shape) and computes their inner product. 


```r
at <- torch$as_tensor(a)
bt <- torch$as_tensor(b)

# torch$dot(at, bt)  <- RuntimeError: dot: Expected 1-D argument self, but got 2-D
# at %.*% bt
```

If we perform the same dot product operation in Python, we get the same error:



```python
import torch
import numpy as np

a = np.array([[1, 2], [3, 4]])
a
#> array([[1, 2],
#>        [3, 4]])
b = np.array([[1, 2], [3, 4]])
b
#> array([[1, 2],
#>        [3, 4]])
np.dot(a, b)
#> array([[ 7, 10],
#>        [15, 22]])
at = torch.as_tensor(a)
bt = torch.as_tensor(b)

at
#> tensor([[1, 2],
#>         [3, 4]])
bt
#> tensor([[1, 2],
#>         [3, 4]])
torch.dot(at, bt)
#> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: dot: Expected 1-D argument self, but got 2-D
#> 
#> Detailed traceback: 
#>   File "<string>", line 1, in <module>
```



```r
a <- torch$Tensor(list(list(1, 2), list(3, 4)))
b <- torch$Tensor(c(c(1, 2), c(3, 4)))
c <- torch$Tensor(list(list(11, 12), list(13, 14)))

a
#> tensor([[1., 2.],
#>         [3., 4.]])
b
#> tensor([1., 2., 3., 4.])
torch$dot(a, b)
#> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: dot: Expected 1-D argument self, but got 2-D

# this is another way of performing dot product in PyTorch
# a$dot(a)
```


```r
o1 <- torch$ones(2L, 2L)
o2 <- torch$ones(2L, 2L)

o1
#> tensor([[1., 1.],
#>         [1., 1.]])
o2
#> tensor([[1., 1.],
#>         [1., 1.]])

torch$dot(o1, o2)
#> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: dot: Expected 1-D argument self, but got 2-D
o1$dot(o2)
#> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: dot: Expected 1-D argument self, but got 2-D
```



```r
# 1D tensors work fine
r = torch$dot(torch$Tensor(list(4L, 2L, 4L)), torch$Tensor(list(3L, 4L, 1L)))
r
#> tensor(24.)
```


```r
## mm and matmul seem to address the dot product we are looking for in tensors
a = torch$randn(2L, 3L)
b = torch$randn(3L, 4L)

a$mm(b)
#> tensor([[ 0.0949,  0.3683, -0.3098, -2.6119],
#>         [ 0.6810, -0.8565, -2.2415,  0.1681]])
a$matmul(b)
#> tensor([[ 0.0949,  0.3683, -0.3098, -2.6119],
#>         [ 0.6810, -0.8565, -2.2415,  0.1681]])
```

Here is agood explanation: https://stackoverflow.com/a/44525687/5270873


```r
abt <- torch$mm(a, b)$transpose(dim0=0L, dim1=1L)
abt
#> tensor([[ 0.0949,  0.6810],
#>         [ 0.3683, -0.8565],
#>         [-0.3098, -2.2415],
#>         [-2.6119,  0.1681]])
```


```r
at <- a$transpose(dim0=0L, dim1=1L)
bt <- b$transpose(dim0=0L, dim1=1L)

btat <- torch$matmul(bt, at)
btat
#> tensor([[ 0.0949,  0.6810],
#>         [ 0.3683, -0.8565],
#>         [-0.3098, -2.2415],
#>         [-2.6119,  0.1681]])
```

$$(A B)^T = B^T A^T$$


```r
# tolerance
torch$allclose(abt, btat, rtol=0.0001)
#> [1] TRUE
```

