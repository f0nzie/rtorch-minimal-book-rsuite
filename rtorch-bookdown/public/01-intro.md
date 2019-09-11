
# Introduction {#intro}

## Motivation
_Why do we want a package of something that is already working well, such as PyTorch?_

There are several reasons, but the main one is to bring another machine learning framework to R. Probably it just me but I feel PyTorch very comfortable to work with. Feels pretty much like everything else in Python. I have tried other frameworks in R. The closest that matches a natural language like PyTorch, is [MXnet](). Unfortunately, it is the hardest to install and maintain after updates. 

Yes. I could have worked directly with PyTorch in a native Python environment, such as Jupyter or PyCharm but it very hard to quit __RMarkdown__ once you get used to it. It is the real thing in regards to [literate programming](https://en.wikipedia.org/wiki/Literate_programming). It does not only contributes to improving the quality of the code but establishes a workflow for a better understanding of the subject by your intended readers [@knuth1983], in what is been called the _literate programming paradigm_ [@cordes1991].

This has the additional benefit of giving the ability to write combination of Python and R code together in the same document. There will times when it is better to create a class in Python; and other times where R will be more convenient to handle a data structure.



## How do we start using `rTorch`
Start using `rTorch` is very simple. After installing the minimum system requirements, you just call it with:


```r
library(rTorch)

transforms  <- torchvision$transforms
```

There are several way of testing that `rTorch` is up and running. Let's see some of them:

### Getting the PyTorch version


```r
rTorch::torch_version()
#> [1] "1.1"
```

### PyTorch configuration
This will show the PyTorch version and the current version of Python installed, as well as the paths where they reside.


```r
rTorch::torch_config()
#> PyTorch v1.1.0 (~/anaconda3/envs/r-torch/lib/python3.6/site-packages/torch)
#> Python v3.6 (~/anaconda3/envs/r-torch/bin/python)
```



## What can you do with `rTorch`
Practically, you can do everything you could with PyTorch within the R ecosystem.
Additionally to the `rTorch` module, from where you can extract methods, functions and classes, there are available two more modules: `torchvision` and `np`, which is short for `numpy`.

### The `torchvision` module
This is an example of using the `torchvision` module. With `torchvision` we could download any of the datasets available. In this case, we will be downloading the training dataset of the **MNIST** handwritten digits. There are 60,000 images in the training set.


```r
local_folder <- '../datasets/mnist_digits'
train_dataset = torchvision$datasets$MNIST(root = local_folder, 
                                           train = TRUE, 
                                           transform = transforms$ToTensor(),
                                           download = TRUE)

train_dataset
#> Dataset MNIST
#>     Number of datapoints: 60000
#>     Root location: ../datasets/mnist_digits
#>     Split: Train
```

You can do similarly for the `test` dataset if you set the flag `train = FALSE`. The `test` dataset has only 10,000 images.


```r
test_dataset = torchvision$datasets$MNIST(root = local_folder, 
                                          train = FALSE, 
                                          transform = transforms$ToTensor())
test_dataset
#> Dataset MNIST
#>     Number of datapoints: 10000
#>     Root location: ../datasets/mnist_digits
#>     Split: Test
```


### `np`: the `numpy` module
`numpy` is automaticaly installed when `PyTorch` is. There is some interdependence between both.  Anytime that we need to do some transformation that is not available in `PyTorch`, we will use `numpy`.

There are several operations that we could perform with `numpy`:

#### Create an array {-}


```r
# do some array manipulations with NumPy
a <- np$array(c(1:4))
a
#> [1] 1 2 3 4
```


```r
np$reshape(np$arange(0, 9), c(3L, 3L))
#>      [,1] [,2] [,3]
#> [1,]    0    1    2
#> [2,]    3    4    5
#> [3,]    6    7    8
```


```r
np$array(list(
             list(73, 67, 43),
             list(87, 134, 58),
             list(102, 43, 37),
             list(73, 67, 43), 
             list(91, 88, 64), 
             list(102, 43, 37), 
             list(69, 96, 70), 
             list(91, 88, 64), 
             list(102, 43, 37), 
             list(69, 96, 70)
           ), dtype='float32')
#>       [,1] [,2] [,3]
#>  [1,]   73   67   43
#>  [2,]   87  134   58
#>  [3,]  102   43   37
#>  [4,]   73   67   43
#>  [5,]   91   88   64
#>  [6,]  102   43   37
#>  [7,]   69   96   70
#>  [8,]   91   88   64
#>  [9,]  102   43   37
#> [10,]   69   96   70
```


#### Reshape an array {-}
For the same `test` dataset that we loaded above, we will show the image of the handwritten digit and its label or class. Before plotting the image, we need to:

1. Extract the image and label from the dataset
2. Convert the tensor to a numpy array
3. Reshape the tensor as a 2D array
4. Plot the digit and its label



```r
rotate <- function(x) t(apply(x, 2, rev))   # function to rotate the matrix

# label for the image
label <- test_dataset[0][[2]]
label    
#> [1] 7

# convert tensor to numpy array
.show_img <- test_dataset[0][[1]]$numpy()
dim(.show_img) 
#> [1]  1 28 28

# reshape 3D array to 2D 
show_img <- np$reshape(.show_img, c(28L, 28L))
dim(show_img)
#> [1] 28 28
```


```r
# show in grays and rotate
image(rotate(show_img), col = gray.colors(64))
title(label)
```

<img src="01-intro_files/figure-html/unnamed-chunk-10-1.png" width="70%" style="display: block; margin: auto;" />


#### Generate a random array {-}


```r
# set the seed
np$random$seed(123L)
# generate a random array
x = np$random$rand(100L)
# calculate the y array
y = np$sin(x) * np$power(x, 3L) + 3L * x + np$random$rand(100L) * 0.8

plot(x, y)
```

<img src="01-intro_files/figure-html/unnamed-chunk-11-1.png" width="70%" style="display: block; margin: auto;" />

#### Convert a `numpy` array to a PyTorch tensor {-}
This is a very common operation that I have seen in examples using PyTorch. Creating the array in `numpy`. and then convert it to a tensor.


```r
# input array
x = np$array(rbind(
            c(0,0,1),
            c(0,1,1),
            c(1,0,1),
            c(1,1,1)))

# the numpy array
x
#>      [,1] [,2] [,3]
#> [1,]    0    0    1
#> [2,]    0    1    1
#> [3,]    1    0    1
#> [4,]    1    1    1
```


```r
# convert the numpy array to float
X <- np$float32(x)
# convert the numpy array to a float tensor
X <- torch$FloatTensor(X)
X
#> tensor([[0., 0., 1.],
#>         [0., 1., 1.],
#>         [1., 0., 1.],
#>         [1., 1., 1.]])
```

### Python built-in functions
To access the Python built-in functions we make use of the package `reticulate` and the function `import_builtins()`.


```r
py_bi <- import_builtins()
```


#### Length of a dataset {-}


```r
py_bi$len(train_dataset)
#> [1] 60000
py_bi$len(test_dataset)
#> [1] 10000
```

#### Iterators {-}


```r
# iterate through training dataset
enum_train_dataset <- py_bi$enumerate(train_dataset)

cat(sprintf("%8s %8s \n", "index", "label"))
#>    index    label
for (i in 1:py_bi$len(train_dataset)) {
    obj <- reticulate::iter_next(enum_train_dataset)
    idx   <- obj[[1]]        # index number
    cat(sprintf("%8d %5d \n", idx, obj[[2]][[2]]))
    
    if (i >= 100) break   # print only 100 labels
}
#>        0     5 
#>        1     0 
#>        2     4 
#>        3     1 
#>        4     9 
#>        5     2 
#>        6     1 
#>        7     3 
#>        8     1 
#>        9     4 
#>       10     3 
#>       11     5 
#>       12     3 
#>       13     6 
#>       14     1 
#>       15     7 
#>       16     2 
#>       17     8 
#>       18     6 
#>       19     9 
#>       20     4 
#>       21     0 
#>       22     9 
#>       23     1 
#>       24     1 
#>       25     2 
#>       26     4 
#>       27     3 
#>       28     2 
#>       29     7 
#>       30     3 
#>       31     8 
#>       32     6 
#>       33     9 
#>       34     0 
#>       35     5 
#>       36     6 
#>       37     0 
#>       38     7 
#>       39     6 
#>       40     1 
#>       41     8 
#>       42     7 
#>       43     9 
#>       44     3 
#>       45     9 
#>       46     8 
#>       47     5 
#>       48     9 
#>       49     3 
#>       50     3 
#>       51     0 
#>       52     7 
#>       53     4 
#>       54     9 
#>       55     8 
#>       56     0 
#>       57     9 
#>       58     4 
#>       59     1 
#>       60     4 
#>       61     4 
#>       62     6 
#>       63     0 
#>       64     4 
#>       65     5 
#>       66     6 
#>       67     1 
#>       68     0 
#>       69     0 
#>       70     1 
#>       71     7 
#>       72     1 
#>       73     6 
#>       74     3 
#>       75     0 
#>       76     2 
#>       77     1 
#>       78     1 
#>       79     7 
#>       80     9 
#>       81     0 
#>       82     2 
#>       83     6 
#>       84     7 
#>       85     8 
#>       86     3 
#>       87     9 
#>       88     0 
#>       89     4 
#>       90     6 
#>       91     7 
#>       92     4 
#>       93     6 
#>       94     8 
#>       95     0 
#>       96     7 
#>       97     8 
#>       98     3 
#>       99     1
```


#### Types and instances {-}


```r
# get the class of the object
py_bi$type(train_dataset)
#> <class 'torchvision.datasets.mnist.MNIST'>

# is train_dataset a torchvision dataset class
py_bi$isinstance(train_dataset, torchvision$datasets$mnist$MNIST)
#> [1] TRUE
```


