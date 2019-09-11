
# rTorch and PyTorch: What's different
This chapter will explain the main differences between `PyTorch` and `rTorch`.
Most of the things work directly in `PyTorch` but we need to be aware of some minor differences when working with rTorch.

Here is a review of existing methods.


```r
library(rTorch)
```

## Calling objects from PyTorch
We use the dollar sign or `$` to call a class, function or method from the rTorch modules. In this case, from `torch` module:


```r
torch$tensor
#> <built-in method tensor of type>
```


## Call a module from PyTorch


```r
# these are the equivalents of import module
nn          <- torch$nn
transforms  <- torchvision$transforms
dsets       <- torchvision$datasets
```

Then we can proceed to extract classes, methods and functions from the `nn`, `transforms`, and `dsets` objects.

## Show the attributes (methods) of a class or PyTorch object
Sometimes we are interested in knowing the internal components of a class. In that case, we use the reticulate function `py_list_attributes()`.


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


```r
reticulate::py_list_attributes(train_dataset)
#>  [1] "__add__"                "__class__"             
#>  [3] "__delattr__"            "__dict__"              
#>  [5] "__dir__"                "__doc__"               
#>  [7] "__eq__"                 "__format__"            
#>  [9] "__ge__"                 "__getattribute__"      
#> [11] "__getitem__"            "__gt__"                
#> [13] "__hash__"               "__init__"              
#> [15] "__init_subclass__"      "__le__"                
#> [17] "__len__"                "__lt__"                
#> [19] "__module__"             "__ne__"                
#> [21] "__new__"                "__reduce__"            
#> [23] "__reduce_ex__"          "__repr__"              
#> [25] "__setattr__"            "__sizeof__"            
#> [27] "__str__"                "__subclasshook__"      
#> [29] "__weakref__"            "_check_exists"         
#> [31] "_format_transform_repr" "_repr_indent"          
#> [33] "class_to_idx"           "classes"               
#> [35] "data"                   "download"              
#> [37] "extra_repr"             "extract_gzip"          
#> [39] "processed_folder"       "raw_folder"            
#> [41] "root"                   "target_transform"      
#> [43] "targets"                "test_data"             
#> [45] "test_file"              "test_labels"           
#> [47] "train"                  "train_data"            
#> [49] "train_labels"           "training_file"         
#> [51] "transform"              "transforms"            
#> [53] "urls"
```

Knowing the internal methods of a class could be useful when we want to refer to a specific property of such class. For example, from the list above, we know that the object `train_dataset` has an attribute `__len__`. We can call it like this:


```r
train_dataset$`__len__`()
#> [1] 60000
```


## Enumeration


```r

x_train = array(c(3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 
                  9.779, 6.182, 7.59, 2.167, 7.042,
                  10.791, 5.313, 7.997, 3.1), dim = c(15,1))

x_train <- r_to_py(x_train)
x_train <- torch$from_numpy(x_train)         # convert to tensor
x_train <- x_train$type(torch$FloatTensor)   # make it a a FloatTensor

x_train
#> tensor([[ 3.3000],
#>         [ 4.4000],
#>         [ 5.5000],
#>         [ 6.7100],
#>         [ 6.9300],
#>         [ 4.1680],
#>         [ 9.7790],
#>         [ 6.1820],
#>         [ 7.5900],
#>         [ 2.1670],
#>         [ 7.0420],
#>         [10.7910],
#>         [ 5.3130],
#>         [ 7.9970],
#>         [ 3.1000]])
```



```r
x_train$nelement()    # number of elements in the tensor
#> [1] 15
```


## How to iterate

### Using `enumerate` and `iterate`


```r
py = import_builtins()

enum_x_train = py$enumerate(x_train)
enum_x_train
#> <enumerate>

py$len(x_train)
#> [1] 15
```



```r
xit = iterate(enum_x_train, simplify = TRUE)
xit
#> [[1]]
#> [[1]][[1]]
#> [1] 0
#> 
#> [[1]][[2]]
#> tensor([3.3000])
#> 
#> 
#> [[2]]
#> [[2]][[1]]
#> [1] 1
#> 
#> [[2]][[2]]
#> tensor([4.4000])
#> 
#> 
#> [[3]]
#> [[3]][[1]]
#> [1] 2
#> 
#> [[3]][[2]]
#> tensor([5.5000])
#> 
#> 
#> [[4]]
#> [[4]][[1]]
#> [1] 3
#> 
#> [[4]][[2]]
#> tensor([6.7100])
#> 
#> 
#> [[5]]
#> [[5]][[1]]
#> [1] 4
#> 
#> [[5]][[2]]
#> tensor([6.9300])
#> 
#> 
#> [[6]]
#> [[6]][[1]]
#> [1] 5
#> 
#> [[6]][[2]]
#> tensor([4.1680])
#> 
#> 
#> [[7]]
#> [[7]][[1]]
#> [1] 6
#> 
#> [[7]][[2]]
#> tensor([9.7790])
#> 
#> 
#> [[8]]
#> [[8]][[1]]
#> [1] 7
#> 
#> [[8]][[2]]
#> tensor([6.1820])
#> 
#> 
#> [[9]]
#> [[9]][[1]]
#> [1] 8
#> 
#> [[9]][[2]]
#> tensor([7.5900])
#> 
#> 
#> [[10]]
#> [[10]][[1]]
#> [1] 9
#> 
#> [[10]][[2]]
#> tensor([2.1670])
#> 
#> 
#> [[11]]
#> [[11]][[1]]
#> [1] 10
#> 
#> [[11]][[2]]
#> tensor([7.0420])
#> 
#> 
#> [[12]]
#> [[12]][[1]]
#> [1] 11
#> 
#> [[12]][[2]]
#> tensor([10.7910])
#> 
#> 
#> [[13]]
#> [[13]][[1]]
#> [1] 12
#> 
#> [[13]][[2]]
#> tensor([5.3130])
#> 
#> 
#> [[14]]
#> [[14]][[1]]
#> [1] 13
#> 
#> [[14]][[2]]
#> tensor([7.9970])
#> 
#> 
#> [[15]]
#> [[15]][[1]]
#> [1] 14
#> 
#> [[15]][[2]]
#> tensor([3.1000])
```


### Using a `for-loop` to iterate


```r
# reset the iterator
enum_x_train = py$enumerate(x_train)

for (i in 1:py$len(x_train)) {
    obj <- iter_next(enum_x_train)    # next item
    cat(obj[[1]], "\t")     # 1st part or index
    print(obj[[2]])         # 2nd part or tensor
}
#> 0 	tensor([3.3000])
#> 1 	tensor([4.4000])
#> 2 	tensor([5.5000])
#> 3 	tensor([6.7100])
#> 4 	tensor([6.9300])
#> 5 	tensor([4.1680])
#> 6 	tensor([9.7790])
#> 7 	tensor([6.1820])
#> 8 	tensor([7.5900])
#> 9 	tensor([2.1670])
#> 10 	tensor([7.0420])
#> 11 	tensor([10.7910])
#> 12 	tensor([5.3130])
#> 13 	tensor([7.9970])
#> 14 	tensor([3.1000])
```

> We will find very frequently this kind of iterators when we read a dataset using `torchvision`. There are different ways to iterate through these objects.

## Zero gradient
The zero gradient was one of the most difficult to implement in R if we don't pay attention to the content of the objects carrying the weights and biases. This happens when the algorithm written in PyTorch is not immediately translatable to rTorch. This can be appreciated in this example.

### Version in Python


```python
import numpy as np
import torch

torch.manual_seed(0)  # reproducible

# Input (temp, rainfall, humidity)
#> <torch._C.Generator object at 0x7f653dd8edf0>
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')
                    

# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# random weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# function for the model
def model(x):
  wt = w.t()
  mm = x @ w.t()
  return x @ w.t() + b       # @ represents matrix multiplication in PyTorch

# MSE loss function
def mse(t1, t2):
  diff = t1 - t2
  return torch.sum(diff * diff) / diff.numel()

# Running all together
# Train for 100 epochs
for i in range(100):
  preds = model(inputs)
  loss = mse(preds, targets)
  loss.backward()
  with torch.no_grad():
    w -= w.grad * 0.00001
    b -= b.grad * 0.00001
    w_gz = w.grad.zero_()
    b_gz = b.grad.zero_()
    
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print("Loss: ", loss)    

# predictions
#> Loss:  tensor(1270.1234, grad_fn=<DivBackward0>)
print("\nPredictions:")
#> 
#> Predictions:
preds

# Targets
#> tensor([[ 69.3122,  80.2639],
#>         [ 73.7528,  97.2381],
#>         [118.3933, 124.7628],
#>         [ 89.6111,  93.0286],
#>         [ 47.3014,  80.6467]], grad_fn=<AddBackward0>)
print("\nTargets:")
#> 
#> Targets:
targets
#> tensor([[ 56.,  70.],
#>         [ 81., 101.],
#>         [119., 133.],
#>         [ 22.,  37.],
#>         [103., 119.]])
```

### Version in R


```r
library(rTorch)

torch$manual_seed(0)
#> <torch._C.Generator>

device = torch$device('cpu')
# Input (temp, rainfall, humidity)
inputs = np$array(list(list(73, 67, 43),
                   list(91, 88, 64),
                   list(87, 134, 58),
                   list(102, 43, 37),
                   list(69, 96, 70)), dtype='float32')

# Targets (apples, oranges)
targets = np$array(list(list(56, 70), 
                    list(81, 101),
                    list(119, 133),
                    list(22, 37), 
                    list(103, 119)), dtype='float32')


# Convert inputs and targets to tensors
inputs = torch$from_numpy(inputs)
targets = torch$from_numpy(targets)

# random numbers for weights and biases. Then convert to double()
torch$set_default_dtype(torch$float64)

w = torch$randn(2L, 3L, requires_grad=TRUE) #$double()
b = torch$randn(2L, requires_grad=TRUE) #$double()

model <- function(x) {
  wt <- w$t()
  return(torch$add(torch$mm(x, wt), b))
}

# MSE loss
mse = function(t1, t2) {
  diff <- torch$sub(t1, t2)
  mul <- torch$sum(torch$mul(diff, diff))
  return(torch$div(mul, diff$numel()))
}

# Running all together
# Adjust weights and reset gradients
for (i in 1:100) {
  preds = model(inputs)
  loss = mse(preds, targets)
  loss$backward()
  with(torch$no_grad(), {
    w$data <- torch$sub(w$data, torch$mul(w$grad, torch$scalar_tensor(1e-5)))
    b$data <- torch$sub(b$data, torch$mul(b$grad, torch$scalar_tensor(1e-5)))
    
    w$grad$zero_()
    b$grad$zero_()
  })
}

# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
cat("Loss: "); print(loss)
#> Loss:
#> tensor(1270.1237, grad_fn=<DivBackward0>)

# predictions
cat("\nPredictions:\n")
#> 
#> Predictions:
preds
#> tensor([[ 69.3122,  80.2639],
#>         [ 73.7528,  97.2381],
#>         [118.3933, 124.7628],
#>         [ 89.6111,  93.0286],
#>         [ 47.3013,  80.6467]], grad_fn=<AddBackward0>)

# Targets
cat("\nTargets:\n")
#> 
#> Targets:
targets
#> tensor([[ 56.,  70.],
#>         [ 81., 101.],
#>         [119., 133.],
#>         [ 22.,  37.],
#>         [103., 119.]])
```

Notice that while in Python, the tensor operation, gradient of the weights times the **Learning Rate**, is:

$$w = -w + \nabla w \; \alpha$$

is a very straight forwward and clean code:


```python
w -= w.grad * 1e-5
```

In R shows a little bit more convoluted:


```r
w$data <- torch$sub(w$data, torch$mul(w$grad, torch$scalar_tensor(1e-5)))
```


## Transform a tensor
Explain how transform a tensor back and forth to `numpy`.
Why is this important?


## Build a model class
PyTorch classes cannot not directly instantiated from `R`. We need an intermediate step to create a class. For this, we use `reticulate` functions that will read the class implementation in `Python` code.


### Example 1

### Example 2


## Convert a tensor to `numpy` object
This is a frequent operation. I have found that this is necessary when 

* a numpy function is not implemented in PyTorch
* We need to convert a tensor to R
* Perform a Boolean operation that is not directly available in PyTorch.


## Convert a `numpy` object to an `R` object
This is mainly required for these reasons:

1. Create a data structure in R
2. Plot using r-base or ggplot2
3. Perform an analysis on parts of a tensor
4. Use R statistical functions that are not available in PyTorch.



