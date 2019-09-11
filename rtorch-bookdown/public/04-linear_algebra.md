
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
<<<<<<< Updated upstream
#> tensor([[[7.6532e-01, 8.9603e-01, 5.1156e-01,  ..., 7.9406e-01,
#>           7.8497e-01, 8.4941e-01],
#>          [9.1182e-01, 6.5170e-01, 9.2817e-02,  ..., 3.4771e-01,
#>           9.4041e-01, 9.4110e-01],
#>          [6.0806e-02, 3.0907e-01, 5.5305e-01,  ..., 5.9411e-01,
#>           1.1907e-01, 7.1095e-01],
#>          ...,
#>          [8.3975e-01, 2.0018e-01, 8.6668e-01,  ..., 8.8767e-01,
#>           7.7704e-01, 4.5707e-01],
#>          [9.4214e-01, 7.5683e-02, 9.8126e-02,  ..., 4.7521e-01,
#>           7.3121e-01, 2.1155e-01],
#>          [5.0467e-01, 3.4027e-01, 7.2210e-01,  ..., 6.5715e-01,
#>           4.8397e-01, 2.8620e-01]],
#> 
#>         [[7.5030e-01, 4.2541e-01, 9.3962e-01,  ..., 4.9645e-01,
#>           9.8028e-01, 4.4452e-01],
#>          [7.0427e-01, 2.0173e-01, 9.2094e-01,  ..., 2.8456e-01,
#>           2.5925e-01, 7.5959e-01],
#>          [7.1519e-01, 3.7549e-01, 1.1528e-01,  ..., 3.3519e-01,
#>           8.3293e-01, 8.4687e-01],
#>          ...,
#>          [7.8535e-04, 9.9786e-01, 9.6608e-02,  ..., 7.6740e-01,
#>           9.7326e-01, 5.2287e-01],
#>          [6.4920e-01, 3.6560e-01, 4.0093e-01,  ..., 8.8842e-01,
#>           9.6601e-01, 7.0590e-01],
#>          [8.8334e-01, 9.2748e-01, 6.1862e-01,  ..., 4.4273e-01,
#>           1.9844e-01, 7.2815e-02]],
#> 
#>         [[6.3738e-01, 8.7195e-01, 9.1957e-01,  ..., 3.1964e-02,
#>           2.1991e-01, 8.8142e-01],
#>          [5.6655e-01, 4.3366e-01, 2.9997e-01,  ..., 8.9414e-01,
#>           7.0536e-01, 9.1174e-01],
#>          [1.6505e-01, 3.4025e-01, 8.1119e-01,  ..., 6.2652e-01,
#>           4.2713e-01, 1.9843e-01],
#>          ...,
#>          [6.3347e-01, 4.2853e-01, 6.9190e-01,  ..., 5.7643e-01,
#>           8.4536e-01, 4.7700e-01],
#>          [1.4390e-02, 8.2787e-01, 5.8873e-01,  ..., 1.9240e-01,
#>           7.0468e-01, 9.7810e-01],
#>          [4.5573e-01, 8.3521e-01, 8.4846e-01,  ..., 3.7869e-01,
#>           9.0218e-02, 1.9664e-01]]])
=======
#> tensor([[[0.6684, 0.5528, 0.0270,  ..., 0.2908, 0.6049, 0.8915],
#>          [0.0394, 0.3462, 0.5272,  ..., 0.4457, 0.5976, 0.7838],
#>          [0.7382, 0.0207, 0.7394,  ..., 0.8856, 0.5818, 0.3219],
#>          ...,
#>          [0.4024, 0.7728, 0.3091,  ..., 0.7930, 0.6596, 0.7067],
#>          [0.7538, 0.4316, 0.2836,  ..., 0.1152, 0.8164, 0.5990],
#>          [0.9225, 0.7821, 0.0251,  ..., 0.7454, 0.1878, 0.6371]],
#> 
#>         [[0.8403, 0.3976, 0.8570,  ..., 0.5182, 0.4880, 0.5795],
#>          [0.0119, 0.1188, 0.5651,  ..., 0.3061, 0.4639, 0.0038],
#>          [0.0646, 0.8077, 0.0327,  ..., 0.4452, 0.7924, 0.5819],
#>          ...,
#>          [0.1694, 0.3516, 0.7494,  ..., 0.8072, 0.3206, 0.4698],
#>          [0.2269, 0.1268, 0.4128,  ..., 0.2996, 0.6468, 0.2020],
#>          [0.3820, 0.9332, 0.1759,  ..., 0.5996, 0.8933, 0.9323]],
#> 
#>         [[0.4245, 0.7981, 0.6913,  ..., 0.8755, 0.6339, 0.1561],
#>          [0.1474, 0.7461, 0.4654,  ..., 0.3012, 0.9493, 0.6266],
#>          [0.9905, 0.7093, 0.3708,  ..., 0.6958, 0.2993, 0.1327],
#>          ...,
#>          [0.2872, 0.4934, 0.0750,  ..., 0.3681, 0.4192, 0.7496],
#>          [0.9082, 0.7270, 0.3292,  ..., 0.8776, 0.5453, 0.2913],
#>          [0.3777, 0.0671, 0.3414,  ..., 0.9847, 0.4750, 0.5425]]])
>>>>>>> Stashed changes
img$shape
#> torch.Size([3, 28, 28])
```


```r
img[1, 1, 1]
<<<<<<< Updated upstream
#> tensor(0.7653)
img[3, 28, 28]
#> tensor(0.1966)
=======
#> tensor(0.6684)
img[3, 28, 28]
#> tensor(0.5425)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
#> tensor([[-1.0737,  0.6377, -0.6390, -1.1473],
#>         [ 0.5132, -2.2162, -1.0623,  0.7173]])
a$matmul(b)
#> tensor([[-1.0737,  0.6377, -0.6390, -1.1473],
#>         [ 0.5132, -2.2162, -1.0623,  0.7173]])
=======
#> tensor([[-0.6183, -0.5196, -0.2773,  1.3294],
#>         [ 0.1229,  0.0799, -0.0070, -2.5501]])
a$matmul(b)
#> tensor([[-0.6183, -0.5196, -0.2773,  1.3294],
#>         [ 0.1229,  0.0799, -0.0070, -2.5501]])
>>>>>>> Stashed changes
```

Here is agood explanation: https://stackoverflow.com/a/44525687/5270873


```r
abt <- torch$mm(a, b)$transpose(dim0=0L, dim1=1L)
abt
<<<<<<< Updated upstream
#> tensor([[-1.0737,  0.5132],
#>         [ 0.6377, -2.2162],
#>         [-0.6390, -1.0623],
#>         [-1.1473,  0.7173]])
=======
#> tensor([[-0.6183,  0.1229],
#>         [-0.5196,  0.0799],
#>         [-0.2773, -0.0070],
#>         [ 1.3294, -2.5501]])
>>>>>>> Stashed changes
```


```r
at <- a$transpose(dim0=0L, dim1=1L)
bt <- b$transpose(dim0=0L, dim1=1L)

btat <- torch$matmul(bt, at)
btat
<<<<<<< Updated upstream
#> tensor([[-1.0737,  0.5132],
#>         [ 0.6377, -2.2162],
#>         [-0.6390, -1.0623],
#>         [-1.1473,  0.7173]])
=======
#> tensor([[-0.6183,  0.1229],
#>         [-0.5196,  0.0799],
#>         [-0.2773, -0.0070],
#>         [ 1.3294, -2.5501]])
>>>>>>> Stashed changes
```

$$(A B)^T = B^T A^T$$


```r
# tolerance
torch$allclose(abt, btat, rtol=0.0001)
#> [1] TRUE
```

