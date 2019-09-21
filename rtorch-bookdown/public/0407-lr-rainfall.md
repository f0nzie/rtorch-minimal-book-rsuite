
# Rainfall. Linear Regression


```r
library(rTorch)
```


Select the device: CPU or GPU


```r
torch$manual_seed(0)
#> <torch._C.Generator>

device = torch$device('cpu')
```

## Training data
The training data can be represented using 2 matrices (inputs and targets), each with one row per observation, and one column per variable.


```r
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
```


## Convert arrays to tensors
Before we build a model, we need to convert inputs and targets to PyTorch tensors.


```r
# Convert inputs and targets to tensors
inputs = torch$from_numpy(inputs)
targets = torch$from_numpy(targets)

print(inputs)
#> tensor([[ 73.,  67.,  43.],
#>         [ 91.,  88.,  64.],
#>         [ 87., 134.,  58.],
#>         [102.,  43.,  37.],
#>         [ 69.,  96.,  70.]], dtype=torch.float64)
print(targets)
#> tensor([[ 56.,  70.],
#>         [ 81., 101.],
#>         [119., 133.],
#>         [ 22.,  37.],
#>         [103., 119.]], dtype=torch.float64)
```


The weights and biases can also be represented as matrices, initialized with random values. The first row of $w$ and the first element of $b$ are used to predict the first target variable, i.e. yield for apples, and, similarly, the second for oranges.


```r
# random numbers for weights and biases. Then convert to double()
torch$set_default_dtype(torch$double)

w = torch$randn(2L, 3L, requires_grad=TRUE)  #$double()
b = torch$randn(2L, requires_grad=TRUE)      #$double()

print(w)
#> tensor([[ 1.5410, -0.2934, -2.1788],
#>         [ 0.5684, -1.0845, -1.3986]], requires_grad=True)
print(b)
#> tensor([0.4033, 0.8380], requires_grad=True)
```


## Build the model
The model is simply a function that performs a matrix multiplication of the input $x$ and the weights $w$ (transposed), and adds the bias $b$ (replicated for each observation).


```r
model <- function(x) {
  wt <- w$t()
  return(torch$add(torch$mm(x, wt), b))
}
```

## Generate predictions
The matrix obtained by passing the input data to the model is a set of predictions for the target variables.


```r
# Generate predictions
preds = model(inputs)
print(preds)
#> tensor([[  -0.4516,  -90.4691],
#>         [ -24.6303, -132.3828],
#>         [ -31.2192, -176.1530],
#>         [  64.3523,  -39.5645],
#>         [ -73.9524, -161.9560]], grad_fn=<AddBackward0>)
```


```r
# Compare with targets
print(targets)
#> tensor([[ 56.,  70.],
#>         [ 81., 101.],
#>         [119., 133.],
#>         [ 22.,  37.],
#>         [103., 119.]])
```

Because we've started with random weights and biases, the model does not a very good job of predicting the target variables.

## Loss Function

We can compare the predictions with the actual targets, using the following method:

* Calculate the difference between the two matrices (preds and targets).
* Square all elements of the difference matrix to remove negative values.
* Calculate the average of the elements in the resulting matrix.

The result is a single number, known as the mean squared error (MSE).


```r
# MSE loss
mse = function(t1, t2) {
  diff <- torch$sub(t1, t2)
  mul <- torch$sum(torch$mul(diff, diff))
  return(torch$div(mul, diff$numel()))
}

print(mse)
#> function(t1, t2) {
#>   diff <- torch$sub(t1, t2)
#>   mul <- torch$sum(torch$mul(diff, diff))
#>   return(torch$div(mul, diff$numel()))
#> }
```

## Step by step process

### Compute the losses


```r
# Compute loss
loss = mse(preds, targets)
print(loss)
#> tensor(33060.8053, grad_fn=<DivBackward0>)
# 46194
# 33060.8070
```


The resulting number is called the **loss**, because it indicates how bad the model is at predicting the target variables. Lower the loss, better the model.

### Compute Gradients

With PyTorch, we can automatically compute the gradient or derivative of the loss w.r.t. to the weights and biases, because they have `requires_grad` set to True.


```r
# Compute gradients
loss$backward()
```

The gradients are stored in the .grad property of the respective tensors.


```r
# Gradients for weights
print(w)
#> tensor([[ 1.5410, -0.2934, -2.1788],
#>         [ 0.5684, -1.0845, -1.3986]], requires_grad=True)
print(w$grad)
#> tensor([[ -6938.4351,  -9674.6757,  -5744.0206],
#>         [-17408.7861, -20595.9333, -12453.4702]])
```


```r
# Gradients for bias
print(b)
#> tensor([0.4033, 0.8380], requires_grad=True)
print(b$grad)
#> tensor([ -89.3802, -212.1051])
```

A key insight from calculus is that the gradient indicates the rate of change of the loss, or the slope of the loss function w.r.t. the weights and biases.

* If a gradient element is positive:
  * increasing the element's value slightly will increase the loss.
  * decreasing the element's value slightly will decrease the loss.

* If a gradient element is negative,
  * increasing the element's value slightly will decrease the loss.
  * decreasing the element's value slightly will increase the loss.

The increase or decrease is proportional to the value of the gradient.

### Reset the gradients

Finally, we'll reset the gradients to zero before moving forward, because PyTorch accumulates gradients.


```r
# Reset the gradients
w$grad$zero_()
#> tensor([[0., 0., 0.],
#>         [0., 0., 0.]])
b$grad$zero_()
#> tensor([0., 0.])

print(w$grad)
#> tensor([[0., 0., 0.],
#>         [0., 0., 0.]])
print(b$grad)
#> tensor([0., 0.])
```


#### Adjust weights and biases using gradient descent

We'll reduce the loss and improve our model using the gradient descent algorithm, which has the following steps:

1. Generate predictions
2. Calculate the loss
3. Compute gradients w.r.t the weights and biases
4. Adjust the weights by subtracting a small quantity proportional to the gradient
5. Reset the gradients to zero


```r
# Generate predictions
preds = model(inputs)
print(preds)
#> tensor([[  -0.4516,  -90.4691],
#>         [ -24.6303, -132.3828],
#>         [ -31.2192, -176.1530],
#>         [  64.3523,  -39.5645],
#>         [ -73.9524, -161.9560]], grad_fn=<AddBackward0>)
```


```r
# Calculate the loss
loss = mse(preds, targets)
print(loss)
#> tensor(33060.8053, grad_fn=<DivBackward0>)
```



```r
# Compute gradients
loss$backward()

print(w$grad)
#> tensor([[ -6938.4351,  -9674.6757,  -5744.0206],
#>         [-17408.7861, -20595.9333, -12453.4702]])
print(b$grad)
#> tensor([ -89.3802, -212.1051])
```



```r
# Adjust weights and reset gradients
with(torch$no_grad(), {
  print(w); print(b)    # requires_grad attribute remains
  w$data <- torch$sub(w$data, torch$mul(w$grad$data, torch$scalar_tensor(1e-5)))
  b$data <- torch$sub(b$data, torch$mul(b$grad$data, torch$scalar_tensor(1e-5)))

  print(w$grad$data$zero_())
  print(b$grad$data$zero_())
})
#> tensor([[ 1.5410, -0.2934, -2.1788],
#>         [ 0.5684, -1.0845, -1.3986]], requires_grad=True)
#> tensor([0.4033, 0.8380], requires_grad=True)
#> tensor([[0., 0., 0.],
#>         [0., 0., 0.]])
#> tensor([0., 0.])

print(w)
#> tensor([[ 1.6104, -0.1967, -2.1213],
#>         [ 0.7425, -0.8786, -1.2741]], requires_grad=True)
print(b)
#> tensor([0.4042, 0.8401], requires_grad=True)
```


With the new weights and biases, the model should have a lower loss.


```r
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)
#> tensor(23432.4894, grad_fn=<DivBackward0>)
```


## All together: train for multiple epochs
To reduce the loss further, we repeat the process of adjusting the weights and biases using the gradients multiple times. Each iteration is called an **epoch**.



```r
# Running all together
# Adjust weights and reset gradients
num_epochs <- 100

for (i in 1:num_epochs) {
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
print(loss)
#> tensor(1258.0216, grad_fn=<DivBackward0>)

# predictions
preds
#> tensor([[ 69.2462,  80.2082],
#>         [ 73.7183,  97.2052],
#>         [118.5780, 124.9272],
#>         [ 89.2282,  92.7052],
#>         [ 47.4648,  80.7782]], grad_fn=<AddBackward0>)

# Targets
targets
#> tensor([[ 56.,  70.],
#>         [ 81., 101.],
#>         [119., 133.],
#>         [ 22.,  37.],
#>         [103., 119.]])
```

