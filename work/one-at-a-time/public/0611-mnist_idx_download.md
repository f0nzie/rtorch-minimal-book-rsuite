
# R: Digits recognition on IDX images - DeepLearningWizard

Source:  https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_logistic_regression/



```r
library(rTorch)

torch       <- import("torch")
torchvision <- import("torchvision")
nn          <- import("torch.nn")
transforms  <- import("torchvision.transforms")
dsets       <- import("torchvision.datasets")
builtins    <- import_builtins()
np          <- import("numpy")

batch_size_train <-  64L
```


## Load datasets

### Load training dataset


```r
train_dataset = dsets$MNIST(root=file.path(".", 'data'), 
                            train=TRUE, 
                            transform=transforms$ToTensor(),
                            download=TRUE)
                            
train_dataset
#> Dataset MNIST
#>     Number of datapoints: 60000
#>     Root location: ./data
#>     Split: Train
builtins$len(train_dataset)
#> [1] 60000
```

### Introspection

### Class and length of `train_dataset`


```r
# R
class(train_dataset)
#> [1] "torchvision.datasets.mnist.MNIST"         
#> [2] "torchvision.datasets.vision.VisionDataset"
#> [3] "torch.utils.data.dataset.Dataset"         
#> [4] "python.builtin.object"
length(train_dataset)
#> [1] 2
```


```r
# Python
builtins$type(train_dataset)
#> <class 'torchvision.datasets.mnist.MNIST'>
builtins$len(train_dataset)
#> [1] 60000
py_len(train_dataset)
#> [1] 60000
```

> Note that both similar commands produce different results


```r
names(train_dataset)
#>  [1] "class_to_idx"     "classes"          "data"            
#>  [4] "download"         "extra_repr"       "extract_gzip"    
#>  [7] "processed_folder" "raw_folder"       "root"            
#> [10] "target_transform" "targets"          "test_data"       
#> [13] "test_file"        "test_labels"      "train"           
#> [16] "train_data"       "train_labels"     "training_file"   
#> [19] "transform"        "transforms"       "urls"
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



```r
# this is identical to Python len() function
train_dataset$`__len__`()
#> [1] 60000
```


```r
# this is not what we are looking for which is torch.Size([1, 28, 28])
# d0 <- train_dataset$data[0][0]
d0 <- train_dataset$data[1][1]
#> Warning in `[.torch.Tensor`(train_dataset$data, 1): Incorrect number of
#> dimensions supplied. The number of supplied arguments, (not counting any
#> NULL, tf$newaxis or np$newaxis) must match thenumber of dimensions in the
#> tensor, unless an all_dims() was supplied (this will produce an error in
#> the future)
#> Warning in `[.torch.Tensor`(train_dataset$data[1], 1): Incorrect number of
#> dimensions supplied. The number of supplied arguments, (not counting any
#> NULL, tf$newaxis or np$newaxis) must match thenumber of dimensions in the
#> tensor, unless an all_dims() was supplied (this will produce an error in
#> the future)
class(d0)
#> [1] "torch.Tensor"          "torch._C._TensorBase"  "python.builtin.object"
d0$size()
#> torch.Size([28])
```

```
d0 <- train_dataset$data[0][0]
Error: It looks like you might be using 0-based indexing to extract using `[`. The rTorch package now uses 1-based extraction by default. You can switch to the old behavior (0-based extraction) with: options(torch.extract.one_based = FALSE)
```


```r
# this is identical to train_dataset.data.size() in Python
train_dataset$data$size()
#> torch.Size([60000, 28, 28])
```



```r
# this is not a dimension we are looking for either
train_dataset$data[c(1L)][1L]$size()
#> Warning in `[.torch.Tensor`(train_dataset$data, c(1L)): Incorrect number
#> of dimensions supplied. The number of supplied arguments, (not counting any
#> NULL, tf$newaxis or np$newaxis) must match thenumber of dimensions in the
#> tensor, unless an all_dims() was supplied (this will produce an error in
#> the future)
#> Warning in `[.torch.Tensor`(train_dataset$data[c(1L)], 1L): Incorrect
#> number of dimensions supplied. The number of supplied arguments, (not
#> counting any NULL, tf$newaxis or np$newaxis) must match thenumber of
#> dimensions in the tensor, unless an all_dims() was supplied (this will
#> produce an error in the future)
#> torch.Size([28])
```



```r
# py = import_builtins()
enum_train_dataset <- builtins$enumerate(train_dataset)
class(enum_train_dataset)
#> [1] "python.builtin.iterator"  "python.builtin.enumerate"
#> [3] "python.builtin.object"
# enum_train_dataset$`__count__`
reticulate::py_list_attributes(enum_train_dataset)
#>  [1] "__class__"         "__delattr__"       "__dir__"          
#>  [4] "__doc__"           "__eq__"            "__format__"       
#>  [7] "__ge__"            "__getattribute__"  "__gt__"           
#> [10] "__hash__"          "__init__"          "__init_subclass__"
#> [13] "__iter__"          "__le__"            "__lt__"           
#> [16] "__ne__"            "__new__"           "__next__"         
#> [19] "__reduce__"        "__reduce_ex__"     "__repr__"         
#> [22] "__setattr__"       "__sizeof__"        "__str__"          
#> [25] "__subclasshook__"
```


```r
# this is not a number we were expecting
enum_train_dataset$`__sizeof__`()
#> [1] 48
```




```r
train_dataset$data$nelement()  # total number of elements in the tensor
#> [1] 47040000
train_dataset$data$shape       # shape
#> torch.Size([60000, 28, 28])
train_dataset$data$size()      # size
#> torch.Size([60000, 28, 28])
```



```r
# get index, label and image
# the pointer will move forward everytime we run the chunk
obj   <- reticulate::iter_next(enum_train_dataset)
idx   <- obj[[1]]        # index number

image <- obj[[2]][[1]]
label <- obj[[2]][[2]]

cat(idx, label, class(label), "\t")
#> 0 5 integer 	
print(image$size())
#> torch.Size([1, 28, 28])
```

### Introspection training dataset

#### Inspecting a single image

So this is how a single image is represented in numbers. It's actually a 28 pixel x 28 pixel image which is why you would end up with this 28x28 matrix of numbers. 



#### Inspecting training dataset first element of tuple
This means to access the image, you need to access the first element in the tuple.


```r
# Input Matrix
image$size()
#> torch.Size([1, 28, 28])

# A 28x28 sized image of a digit
# torch.Size([1, 28, 28])
```

### MNIST image from training dataset


```r
class(image$numpy())
#> [1] "array"
dim(image$numpy())
#> [1]  1 28 28
```

#### Plot one image


```r
rotate <- function(x) t(apply(x, 2, rev))   #function to rotate the matrix

# read label for digit
label
#> [1] 5

# read tensor for image
# img_tensor_2d <- image[0]
img_tensor_2d <- image[1L]
#> Warning in `[.torch.Tensor`(image, 1L): Incorrect number of dimensions
#> supplied. The number of supplied arguments, (not counting any NULL,
#> tf$newaxis or np$newaxis) must match thenumber of dimensions in the tensor,
#> unless an all_dims() was supplied (this will produce an error in the
#> future)
img_tensor_2d$shape       # shape of the 2D tensor: torch.Size([28, 28])
#> torch.Size([28, 28])

# convert tensor to numpy array
img_mat_2d <- img_tensor_2d$numpy()
dim(img_mat_2d)
#> [1] 28 28

# show digit image
image(rotate(img_mat_2d))
title(label)
```

<img src="0611-mnist_idx_download_files/figure-html/unnamed-chunk-18-1.png" width="70%" style="display: block; margin: auto;" />

### Plot a second image


```r
# iterate to the next tensor
obj <- reticulate::iter_next(enum_train_dataset)   # iterator
idx <- obj[[1]]
img <- obj[[2]][[1]]
lbl <- obj[[2]][[2]]

img_tensor_2d <- img[1]            # get 2D tensor
#> Warning in `[.torch.Tensor`(img, 1): Incorrect number of dimensions
#> supplied. The number of supplied arguments, (not counting any NULL,
#> tf$newaxis or np$newaxis) must match thenumber of dimensions in the tensor,
#> unless an all_dims() was supplied (this will produce an error in the
#> future)
img_mat_2d <- img_tensor_2d$numpy()  # convert to 2D array

# show digit image
image(rotate(img_mat_2d))            # rotate and plot
title(lbl)                         # label as plot title
```

<img src="0611-mnist_idx_download_files/figure-html/unnamed-chunk-19-1.png" width="70%" style="display: block; margin: auto;" />

### Loading the test dataset


```r
test_dataset = dsets$MNIST(root = '../../data', 
                           train=FALSE, 
                           transform=transforms$ToTensor())

py_len(test_dataset)
#> [1] 10000
```

#### Introspection of the test dataset


```r
# we'll get all the attributes of the class
reticulate::py_list_attributes(test_dataset)
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



```r
# get the Python type
builtins$type(test_dataset$`__getitem__`(0L))   # in Python a tuple gets converted to a list
#> <class 'list'>
# in Python: type(test_dataset[0]) -> <class 'tuple'>
```


```r
# the size of the first and last image tensor
test_dataset$`__getitem__`(0L)[[1]]$size()      # same as test_dataset[0][0].size()
#> torch.Size([1, 28, 28])
test_dataset$`__getitem__`(9999L)[[1]]$size()  
#> torch.Size([1, 28, 28])
```

This is the same as:

```r
py_to_r(test_dataset)
#> Dataset MNIST
#>     Number of datapoints: 10000
#>     Root location: ../../data
#>     Split: Test
```



```r
# the size of the first and last image tensor
# py_get_item(test_dataset, 0L)[[1]]$size()
py_get_item(test_dataset, 0L)[[1]]$size()
#> torch.Size([1, 28, 28])
py_get_item(test_dataset, 9999L)[[1]]$size()
#> torch.Size([1, 28, 28])
# same as test_dataset[0][0].size()
```



```r
# the label is the second list member
label <- test_dataset$`__getitem__`(0L)[[2]]  # in Python: test_dataset[0][1]
label
#> [1] 7
```

### Plot image test dataset


```r
# convert tensor to numpy array
.show_img <- test_dataset$`__getitem__`(0L)[[1]]$numpy() 
dim(.show_img)                                           # numpy 3D array
#> [1]  1 28 28

# reshape 3D array to 2D 
show_img <- np$reshape(.show_img, c(28L, 28L))
dim(show_img)
#> [1] 28 28
```


```r
# another way to reshape the array
show_img <- np$reshape(test_dataset$`__getitem__`(0L)[[1]]$numpy(), c(28L, 28L))
dim(show_img)
#> [1] 28 28
```


```r
# show in grays and rotate
image(rotate(show_img), col = gray.colors(64))
title(label)
```

<img src="0611-mnist_idx_download_files/figure-html/unnamed-chunk-29-1.png" width="70%" style="display: block; margin: auto;" />

### Plot a second test image


```r
# next image, index moves from (0L) to (1L), and so on
idx <- 1L
.show_img <- test_dataset$`__getitem__`(idx)[[1]]$numpy()
show_img  <- np$reshape(.show_img, c(28L, 28L))
label     <- test_dataset$`__getitem__`(idx)[[2]]

image(rotate(show_img), col = gray.colors(64))
title(label)
```

<img src="0611-mnist_idx_download_files/figure-html/unnamed-chunk-30-1.png" width="70%" style="display: block; margin: auto;" />

### Plot the last test image


```r
# next image, index moves from (0L) to (1L), and so on
# first image is 0, last image would be 9999
idx <- py_len(test_dataset) - 1L
.show_img <- test_dataset$`__getitem__`(idx)[[1]]$numpy()
show_img  <- np$reshape(.show_img, c(28L, 28L))
label     <- test_dataset$`__getitem__`(idx)[[2]]

image(rotate(show_img), col = gray.colors(64))
title(label)
```

<img src="0611-mnist_idx_download_files/figure-html/unnamed-chunk-31-1.png" width="70%" style="display: block; margin: auto;" />

## Defining epochs
When the model goes through the whole 60k images once, learning how to classify 0-9, it's consider **1 epoch**.

However, there's a concept of batch size where it means the model would look at 100 images before updating the model's weights, thereby learning. When the model updates its weights (parameters) after looking at all the images, this is considered 1 iteration.


```r
batch_size <- 100L
```

We arbitrarily set 3000 iterations here which means the model would update 3000 times. 


```r
n_iters <- 3000L
```

One epoch consists of 60,000 / 100 = 600 iterations. Because we would like to go through 3000 iterations, this implies we would have 3000 / 600 = 5 epochs as each epoch has 600 iterations. 


```r
num_epochs = n_iters / (py_len(train_dataset) / batch_size)
num_epochs = as.integer(num_epochs)
num_epochs
#> [1] 5
```

## Create iterable objects: training and testing dataset


```r
train_loader = torch$utils$data$DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=TRUE)
```



```r
# Iterable object
test_loader = torch$utils$data$DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=FALSE)
```

### Check iteraibility


```r
collections <- import("collections")

builtins$isinstance(train_loader, collections$Iterable)
#> [1] TRUE
builtins$isinstance(test_loader, collections$Iterable)
#> [1] TRUE
```

## Building the model


```r
# Same as linear regression! 
main <- py_run_string(
"
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
")

# build a Linear Rgression model
LogisticRegressionModel <- main$LogisticRegressionModel
```


### Instantiate model class based on input and out dimensions


```r
# feeding the model with 28x28 images
input_dim = 28L*28L

# classify digits 0-9 a total of 10 classes,
output_dim = 10L

model = LogisticRegressionModel(input_dim, output_dim)
```


### Instantiate Cross Entropy Loss class


```r
# need Cross Entropy Loss to calculate loss before we backpropagation
criterion = nn$CrossEntropyLoss()  
```

### Instantiate Optimizer class
Similar to what we've covered above, this calculates the parameters' gradients and update them subsequently. 


```r
# calculate parameters' gradients and update
learning_rate = 0.001

optimizer = torch$optim$SGD(model$parameters(), lr=learning_rate)  
```

### Parameters introspection
You'll realize we have 2 sets of parameters, 10x784 which is $A$ and 10x1 which is $b$ in the $y = AX + b$ equation, where $X$ is our input of size 784.

We'll go into details subsequently how these parameters interact with our input to produce our 10x1 output. 


```r
# Type of parameter object
print(model$parameters())
#> <generator object Module.parameters at 0x7f457b388200>
model_parameters <- builtins$list(model$parameters())

# Length of parameters
print(builtins$len(model_parameters))
#> [1] 2

# FC 1 Parameters 
builtins$list(model_parameters)[[1]]$size()
#> torch.Size([10, 784])

# FC 1 Bias Parameters
builtins$list(model_parameters)[[2]]$size()
#> torch.Size([10])
```


```r
builtins$len(builtins$list(model$parameters()))
#> [1] 2
```

## Train the model and test per epoch


```r
iter = 0

for (epoch in 1:num_epochs) {
  iter_train_dataset <- builtins$enumerate(train_loader) # convert to iterator
  for (obj in iterate(iter_train_dataset)) {
      # get the tensors for images and labels
      images <- obj[[2]][[1]]
      labels <- obj[[2]][[2]]
      
      # Reshape images to (batch_size, input_size)
      images <- images$view(-1L, 28L*28L)$requires_grad_()
      
      # Clear gradients w.r.t. parameters
      optimizer$zero_grad()
      
      # Forward pass to get output/logits
      outputs = model(images)
      
      # Calculate Loss: softmax --> cross entropy loss
      loss = criterion(outputs, labels)
      
      # Getting gradients w.r.t. parameters
      loss$backward()
      
      # Updating parameters
      optimizer$step()
      
      iter = iter + 1
      
      if (iter %% 500 == 0) {
          # Calculate Accuracy         
          correct = 0
          total = 0
          
          # Iterate through test dataset
          iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
          for (obj2 in iterate(iter_test_dataset)) {
              # Load images to a Torch Variable
              images <- obj2[[2]][[1]]
              labels <- obj2[[2]][[2]]
              images <- images$view(-1L, 28L*28L)$requires_grad_()
          
              # Forward pass only to get logits/output
              outputs = model(images)
          
              # Get predictions from the maximum value
              .predicted = torch$max(outputs$data, 1L)
              predicted <- .predicted[1L]
          
              # Total number of labels
              total = total + labels$size(0L)
          
              # Total correct predictions
              correct = correct + sum((predicted$numpy() == labels$numpy()))
          }
          accuracy = 100 * correct / total
          
          # Print Loss
          cat(sprintf('Iteration: %5d. Loss: %f. Accuracy: %8.2f \n', 
                      iter, loss$item(), accuracy))
      }
  }
}  
#> Iteration:   500. Loss: 1.930394. Accuracy:    65.16 
#> Iteration:  1000. Loss: 1.664889. Accuracy:    76.85 
#> Iteration:  1500. Loss: 1.330313. Accuracy:    80.35 
#> Iteration:  2000. Loss: 1.202632. Accuracy:    81.44 
#> Iteration:  2500. Loss: 1.112350. Accuracy:    82.48 
#> Iteration:  3000. Loss: 0.963235. Accuracy:    83.27
```

## Break down accuracy calculation
As we've trained our model, we can extract the accuracy calculation portion to understand what's happening without re-training the model.

This would print out the output of the model's predictions on your notebook.


```r
# Iterate through test dataset
iter_test <- 0
iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
for (test_obj in iterate(iter_test_dataset)) {
    iter_test <- iter_test + 1
    # Load images to a Torch Variable
    images <- test_obj[[2]][[1]]
    labels <- test_obj[[2]][[2]]
    images <- images$view(-1L, 28L*28L)$requires_grad_()
    
    # Forward pass only to get logits/output
    outputs = model(images)
    
    if (iter_test == 1) {
        print('OUTPUTS')
        print(outputs)
    }
    # Get predictions from the maximum value
    .predicted = torch$max(outputs$data, 1L)
    predicted <- .predicted[1L]
}
#> [1] "OUTPUTS"
#> tensor([[-4.4718e-01, -1.2468e+00, -2.0423e-01, -2.0565e-01,  4.9760e-02,
#>          -4.3313e-01, -1.0500e+00,  2.8732e+00, -1.7834e-01,  8.9531e-01],
#>         [ 3.9692e-01,  1.8405e-01,  1.3228e+00,  1.0296e+00, -1.4375e+00,
#>           9.6043e-01,  1.2506e+00, -2.0427e+00,  5.8836e-01, -1.6811e+00],
#>         [-9.9751e-01,  2.1629e+00,  3.9952e-01, -3.6541e-02, -6.3222e-01,
#>          -4.0809e-01, -5.4612e-02, -1.7367e-01,  1.9773e-01, -4.0278e-01],
#>         [ 2.9793e+00, -2.3172e+00, -1.5949e-01, -4.7658e-01, -9.7048e-01,
#>           3.5469e-01,  1.0783e+00,  2.5707e-01, -7.1833e-01, -2.8411e-01],
#>         [-2.2573e-01, -2.1043e+00,  4.9942e-01, -4.8008e-01,  1.7810e+00,
#>          -5.2401e-01,  1.9287e-01,  2.6184e-01,  6.8635e-02,  7.9822e-01],
#>         [-1.4478e+00,  2.7352e+00,  3.2370e-01,  2.1585e-01, -7.5771e-01,
#>          -5.8821e-01, -6.3140e-01,  4.4991e-02,  4.9383e-01, -2.5457e-01],
#>         [-1.0588e+00, -1.2620e+00, -6.5725e-01,  2.6055e-01,  1.4758e+00,
#>           2.3970e-01, -7.7276e-01,  5.3985e-01,  4.3092e-01,  8.9223e-01],
#>         [-1.5584e+00, -1.1613e-01, -7.3851e-01, -1.4457e-01,  7.5339e-01,
#>           3.2954e-01,  4.0229e-01,  2.5227e-02,  7.0832e-02,  1.2668e+00],
#>         [ 3.0124e-01, -5.5646e-01,  8.9757e-01, -1.1597e+00,  4.4120e-01,
#>          -9.9473e-02,  5.4231e-01, -8.7245e-01, -2.4963e-01,  1.4820e-01],
#>         [-5.7911e-01, -8.7807e-01, -9.0174e-01, -1.3825e+00,  9.5467e-01,
#>          -2.4098e-01, -3.9403e-01,  1.6330e+00, -5.8946e-02,  1.7613e+00],
#>         [ 3.1051e+00, -1.9428e+00,  6.6167e-01,  8.6285e-01, -9.0445e-01,
#>           7.9078e-01, -4.4758e-01, -1.4197e+00,  6.4352e-01, -1.9277e+00],
#>         [ 8.0887e-01, -3.3479e-01,  5.9968e-01, -2.2933e-01,  3.9940e-01,
#>          -4.7725e-01,  7.1014e-01, -1.0974e+00,  1.7018e-01, -7.7502e-01],
#>         [-9.1517e-01, -1.5163e+00, -7.1221e-01, -1.2070e+00,  1.4432e+00,
#>          -4.8755e-02, -2.1018e-01,  1.2437e+00,  2.2598e-01,  2.0394e+00],
#>         [ 2.9048e+00, -2.3186e+00, -1.8192e-01, -5.1522e-01, -3.7509e-01,
#>           6.0079e-01, -4.5564e-01, -5.7541e-01,  5.8529e-01, -3.5066e-02],
#>         [-1.3704e+00,  2.7397e+00,  5.9176e-02,  4.3453e-01, -1.1104e+00,
#>          -6.1452e-02, -1.1429e-01, -8.6372e-02,  2.8079e-01, -2.5751e-01],
#>         [ 3.3053e-01, -8.3779e-01, -1.0540e-02,  1.3105e+00, -5.3815e-01,
#>           8.2157e-01, -1.4329e-01, -5.0031e-01,  7.6549e-01, -8.0572e-01],
#>         [-3.5437e-01, -2.1260e+00,  1.8429e-01, -5.7864e-01,  1.4763e+00,
#>          -7.6180e-01, -2.3951e-01,  8.6812e-01,  1.8640e-01,  1.5885e+00],
#>         [ 1.6172e-01, -1.4986e+00, -3.7321e-01,  4.8318e-01, -1.6308e-01,
#>          -3.2916e-01, -7.8972e-01,  2.7884e+00, -4.1788e-01,  5.4859e-01],
#>         [-6.8135e-01, -3.0668e-01, -1.1400e-01,  1.1116e+00, -4.5003e-01,
#>           3.6586e-01,  1.1125e+00, -7.2670e-01,  3.5235e-01, -6.2756e-01],
#>         [-1.1001e+00, -1.3404e+00, -4.0028e-01, -1.2612e-01,  1.9166e+00,
#>           1.4572e-01, -1.2042e-01,  1.4169e-02, -6.7446e-02,  1.4080e+00],
#>         [-9.7837e-01, -3.5000e-01, -1.4249e+00,  2.8612e-02,  5.2230e-01,
#>           2.2429e-01, -1.1744e+00,  1.5846e+00,  1.7682e-01,  1.3263e+00],
#>         [-6.5477e-01, -1.5491e+00, -7.0765e-02,  9.0579e-02,  3.3517e-01,
#>           7.5842e-01,  2.3208e+00, -1.5609e+00,  9.5555e-02,  1.4646e-01],
#>         [-2.6069e-01,  1.1823e-02,  6.3756e-01, -6.9845e-01,  9.5082e-01,
#>          -1.0574e+00,  1.0354e+00, -2.0140e-02, -1.3835e-01, -1.9916e-01],
#>         [ 1.3539e-02, -1.0339e+00, -6.8834e-01,  1.3298e-01, -2.5272e-01,
#>           1.2588e+00,  3.4169e-01, -1.0307e+00,  7.4657e-01, -7.5138e-02],
#>         [-8.6418e-01, -1.0550e+00, -6.2473e-02, -7.9190e-03,  1.3485e+00,
#>          -3.3957e-01, -5.4339e-02,  2.6620e-01, -1.5850e-01,  7.9936e-01],
#>         [ 4.0091e+00, -2.8621e+00,  5.1275e-01, -1.2640e+00, -5.6487e-01,
#>           5.5093e-01,  9.1637e-01, -8.8556e-01, -2.1351e-01, -1.3818e+00],
#>         [-1.4768e-01, -1.2759e+00, -4.6839e-01,  2.0196e-01,  2.9679e-01,
#>          -7.6697e-02, -6.1129e-01,  1.7785e+00, -4.7420e-01,  1.0833e+00],
#>         [-5.7190e-01, -2.3027e+00, -3.3794e-01, -8.1916e-01,  2.2007e+00,
#>           1.3461e-01,  1.8284e-01, -8.1112e-02,  5.7235e-02,  1.4458e+00],
#>         [ 2.7089e+00, -2.3237e+00,  9.4707e-02,  7.7947e-01, -1.2415e+00,
#>           8.0063e-01, -3.9329e-01, -7.2111e-01,  6.3017e-01, -6.2783e-01],
#>         [-1.0973e+00,  1.4323e+00, -2.1174e-01,  1.4088e-01, -4.8525e-01,
#>           3.1773e-01,  2.0053e-01, -9.9906e-02,  2.0743e-01, -2.2311e-01],
#>         [-8.4590e-01, -1.0016e-01, -8.6598e-01,  2.2519e+00, -1.2618e+00,
#>           8.4650e-01, -1.5320e-01,  3.0204e-01,  4.9887e-02, -2.8661e-01],
#>         [-1.0682e+00,  1.0729e+00, -1.1216e-01,  3.4936e-01, -3.4230e-01,
#>           2.2077e-01, -8.2051e-02,  1.2824e-01,  2.2942e-01,  1.1629e-01],
#>         [-9.4899e-01, -5.3057e-01, -5.7026e-01,  2.4066e+00, -3.0957e-01,
#>           1.5446e+00, -8.6426e-02, -1.0682e+00,  5.4839e-01,  3.3472e-02],
#>         [ 1.5498e+00, -1.7359e+00,  6.3691e-01, -1.7320e+00,  7.6195e-01,
#>           1.7419e-01,  1.6346e+00, -8.8359e-01, -2.2536e-01, -2.0428e-01],
#>         [-9.2637e-01, -4.7381e-01,  9.2078e-01, -3.8526e-01, -1.1131e-01,
#>          -7.1767e-01, -1.5457e+00,  1.6373e+00,  5.6451e-01,  5.3076e-01],
#>         [ 2.9096e-01, -1.2399e+00,  2.5253e+00,  1.2543e-01, -9.3077e-01,
#>          -7.8151e-02,  3.7625e-01, -6.0042e-01,  3.1139e-01, -1.6205e+00],
#>         [-5.4234e-01, -1.5158e+00,  1.9148e-01, -6.0352e-02, -1.1139e-01,
#>          -4.1123e-01, -6.5139e-01,  2.3466e+00, -4.2553e-01,  1.1583e+00],
#>         [-1.3526e+00,  2.0814e+00, -2.5008e-01,  2.4792e-02, -5.8163e-01,
#>           6.7936e-02, -1.0535e-01,  1.0695e-01,  2.9362e-01, -8.5246e-03],
#>         [ 1.9078e-01,  5.4378e-01,  8.4019e-01,  1.1426e+00, -1.9290e+00,
#>           3.7091e-01,  5.2896e-01, -1.1999e+00,  6.3426e-01, -1.4869e+00],
#>         [-1.3185e+00,  2.9538e+00, -1.3862e-01,  5.1946e-01, -1.1983e+00,
#>          -5.6529e-02, -1.7690e-01, -4.6838e-01,  6.7856e-01, -3.3835e-01],
#>         [-6.5727e-01,  1.5232e+00,  9.6108e-02,  6.0365e-02, -5.5318e-01,
#>          -1.1610e-01, -5.1669e-02, -1.1898e-01,  1.2403e-01, -2.0924e-01],
#>         [-7.8086e-01, -8.5921e-01, -1.9255e-01, -3.3425e-01,  2.4007e-02,
#>          -2.0580e-01, -6.2357e-01,  2.0483e+00, -2.8002e-01,  1.4973e+00],
#>         [-2.1809e+00, -2.5838e-01, -7.2363e-02, -2.4267e-01,  2.0468e+00,
#>          -7.7054e-01, -8.5603e-01,  6.2093e-01,  3.5206e-01,  1.5995e+00],
#>         [-4.3523e-01,  8.3631e-01,  9.8384e-01,  1.1996e-01, -9.5900e-02,
#>          -4.3901e-01,  3.6827e-01, -1.0706e+00,  4.0128e-01, -6.4364e-01],
#>         [-1.1405e+00,  2.4013e-01,  2.0004e-02,  1.2879e+00, -6.5986e-01,
#>           5.7140e-01,  5.7954e-01, -3.0499e-01, -4.2171e-02, -3.0109e-01],
#>         [ 1.4556e-01, -1.0861e+00, -6.8373e-01,  1.5800e+00, -6.2538e-01,
#>           1.3170e+00,  8.3067e-02, -1.5121e+00,  8.6213e-01, -3.3431e-01],
#>         [-1.7141e+00,  3.6242e-01,  5.5442e-02,  8.1118e-01, -1.2363e-01,
#>           5.2943e-01,  2.3864e-01, -4.6302e-02,  2.1998e-01,  2.6361e-01],
#>         [-6.8859e-01, -1.9947e-01,  1.3936e+00, -4.3002e-01,  4.1049e-01,
#>          -5.2483e-01,  5.7683e-01, -4.7484e-01, -9.8542e-03,  6.1606e-02],
#>         [-1.3215e+00, -2.6090e+00, -1.1861e+00,  2.3431e-02,  2.6590e+00,
#>           3.1867e-01, -6.4302e-01,  2.9892e-01,  6.3076e-01,  2.2543e+00],
#>         [-5.2516e-01, -1.8418e+00,  6.0105e-01, -4.2228e-01,  2.3761e+00,
#>          -9.7343e-01,  2.6593e-01,  1.3026e-01, -3.4686e-01,  1.0964e+00],
#>         [-5.9763e-02, -1.0214e+00,  1.2830e-01,  2.9076e-01, -3.1880e-01,
#>           4.0005e-01,  2.0851e+00, -1.2715e+00,  1.6089e-02, -4.3198e-01],
#>         [ 6.1674e-02, -6.0513e-01, -2.9976e-01,  1.7100e+00, -6.0768e-01,
#>           6.4359e-01,  3.3144e-02, -4.9583e-01, -8.5020e-02, -1.4872e-01],
#>         [ 4.2600e-01, -1.1501e+00, -1.3824e+00,  4.8415e-02,  6.8784e-01,
#>           1.1275e+00,  3.2526e-03, -3.7121e-01, -3.1046e-01,  5.7692e-01],
#>         [ 2.6287e-01, -6.7316e-01, -2.8793e-01,  1.0665e+00, -2.1556e-01,
#>           7.5585e-01, -1.3331e-01, -7.6487e-01,  3.2164e-01, -4.7666e-01],
#>         [ 7.0425e-02, -5.7740e-01,  1.4257e+00, -3.6412e-01,  6.0841e-01,
#>          -8.4833e-01,  6.6877e-01, -7.8570e-01,  1.2101e-01, -3.5506e-01],
#>         [ 1.4769e+00, -2.0905e+00, -7.5154e-01,  2.1655e-01, -5.0935e-01,
#>           1.1331e+00,  2.3215e-01, -9.0899e-01,  1.3230e+00, -8.0555e-02],
#>         [-1.9352e-01, -2.6666e+00,  1.1157e-01, -4.1165e-01,  2.6324e+00,
#>          -1.4759e-01,  3.6568e-01, -3.5276e-01,  4.8593e-02,  7.8187e-01],
#>         [-8.9221e-01,  2.3047e+00,  2.1518e-01,  2.6188e-01, -8.3481e-01,
#>          -3.2789e-01, -4.1367e-01, -4.0623e-02,  3.7693e-01, -2.6286e-01],
#>         [-2.7419e-01, -1.9066e+00, -7.5488e-01, -9.0709e-01,  1.6777e+00,
#>          -3.0612e-01, -1.0424e-01,  1.1124e+00, -2.7138e-01,  2.0957e+00],
#>         [-1.3851e-01,  2.7365e-01, -4.9187e-01, -3.6332e-01,  2.4250e-01,
#>           3.7286e-01, -6.8500e-02,  4.2384e-01,  4.5608e-02, -1.5793e-01],
#>         [-2.3738e-01, -1.4720e+00, -7.5517e-01,  9.6357e-01,  4.9134e-01,
#>           7.1542e-02, -1.9931e-01,  2.1386e+00, -4.6620e-01,  7.3949e-01],
#>         [ 3.8759e-02, -8.3677e-01,  9.2317e-01, -1.6138e+00, -9.7517e-02,
#>           1.8657e-01,  5.3202e-01, -6.8962e-01,  1.1442e+00,  2.5980e-01],
#>         [-8.0620e-01, -9.0926e-01,  1.2432e-01, -2.7728e-01,  7.7447e-01,
#>           1.2153e-01,  1.0216e-01, -5.0359e-02,  4.8220e-01,  9.2443e-01],
#>         [-7.5812e-01, -1.5325e-01,  1.2925e+00,  7.4748e-01,  3.0094e-02,
#>           1.3638e-01, -4.2398e-01, -5.9170e-01,  4.1177e-01,  1.7426e-01],
#>         [-1.0049e+00, -6.0475e-01,  3.7886e-01, -6.3298e-01,  7.2543e-01,
#>          -7.9102e-01, -6.2105e-01,  1.4978e+00,  1.1697e-01,  5.8643e-01],
#>         [-1.1528e+00, -5.6782e-01, -5.2434e-01,  5.1987e-01,  5.7555e-01,
#>           6.6608e-01,  1.5166e-01, -2.7806e-01,  5.9166e-01,  8.2105e-01],
#>         [ 4.1861e-01, -3.5938e-01,  8.0857e-01, -2.0490e-01,  2.2257e-01,
#>          -7.9776e-01,  5.9117e-01,  6.2306e-02, -3.1427e-01, -7.2222e-01],
#>         [-7.1386e-01, -1.5529e+00,  5.5837e-01, -7.6247e-01,  2.0723e+00,
#>          -8.3456e-01, -1.5104e-01,  4.7445e-01,  1.5284e-01,  7.4672e-01],
#>         [-1.4449e+00, -2.0144e-01, -3.0553e-01,  2.7222e+00, -5.0228e-01,
#>           9.1544e-01, -7.1420e-01, -8.1575e-01,  1.0112e+00,  1.0868e-01],
#>         [ 2.4134e+00, -1.4388e+00,  5.2374e-01, -7.8384e-01, -1.4551e+00,
#>           4.6627e-01,  3.9831e-01, -1.2101e-01, -5.1153e-01, -4.7996e-01],
#>         [ 3.3630e-01, -1.1722e+00, -8.2107e-01,  4.1387e-02, -4.6085e-02,
#>          -2.4157e-01, -6.9715e-01,  2.8333e+00, -3.2066e-01,  5.0198e-01],
#>         [ 4.2487e+00, -2.9074e+00,  7.6584e-01,  1.5551e-01, -1.1384e+00,
#>           7.4189e-01,  5.0971e-02, -1.6840e+00,  3.4597e-01, -1.7573e+00],
#>         [ 1.1821e+00, -1.3790e+00,  1.9817e+00,  1.2842e+00, -1.0030e+00,
#>          -4.0824e-02,  3.7487e-01, -1.1415e+00,  3.6272e-01, -1.8491e+00],
#>         [-1.4216e+00,  7.9607e-01,  2.3513e-01, -7.5882e-02, -8.3529e-01,
#>          -5.0411e-01, -1.1313e+00,  7.2496e-01,  1.1474e+00,  5.9806e-01],
#>         [-1.2933e+00,  2.4197e+00, -2.4266e-01,  6.4332e-03, -9.0563e-01,
#>           1.3652e-01, -5.4754e-02, -1.6740e-02,  4.9035e-01, -1.1895e-01],
#>         [-1.9556e+00,  7.0594e-01, -5.8684e-01, -4.6060e-01,  9.5170e-01,
#>          -6.2083e-01, -7.1635e-01,  1.8651e+00,  5.7012e-02,  7.3446e-01],
#>         [-4.6137e-01,  2.4577e-01, -4.2735e-01,  2.3457e+00, -9.3308e-01,
#>           1.0440e+00, -4.7822e-01, -1.0964e+00,  5.1578e-01, -7.8293e-01],
#>         [-8.3095e-01, -3.1335e-01,  6.8333e-01, -6.5840e-01, -1.6188e-01,
#>          -5.4308e-01, -1.0843e-01,  1.5766e+00, -3.0622e-01,  5.3792e-01],
#>         [-1.5588e+00,  9.7683e-01, -5.7430e-01,  1.2733e-01,  1.1985e-02,
#>           1.1834e-01, -5.4369e-01,  3.0420e-01,  8.6740e-01,  7.8307e-01],
#>         [-6.7019e-01, -6.8960e-01, -6.1483e-01, -1.0952e+00, -6.4927e-02,
#>          -4.8818e-01, -1.4076e+00,  3.2018e+00,  2.5664e-01,  9.1204e-01],
#>         [-6.4890e-01, -1.5371e+00, -1.1082e+00,  9.4802e-03,  9.5356e-01,
#>           4.1426e-01, -7.4561e-02,  1.3728e+00, -5.2468e-01,  1.6284e+00],
#>         [-1.4464e-01, -1.9196e+00,  5.0971e-01, -7.0371e-01,  1.9079e-01,
#>           1.7855e-01,  2.2777e+00, -5.5766e-01, -4.1643e-01,  2.2053e-01],
#>         [-7.4571e-01, -1.2401e+00,  3.8547e+00, -2.2660e-03, -2.0006e-01,
#>          -1.2234e+00,  9.5088e-01, -1.0268e+00,  6.0097e-01, -1.1153e+00],
#>         [-4.2298e-01, -1.7918e+00, -4.7802e-01,  9.2942e-02,  8.0780e-01,
#>          -1.5640e-01, -7.1729e-01,  2.1380e+00, -2.9940e-01,  1.5633e+00],
#>         [-7.9096e-01,  1.2363e-01, -4.1729e-01, -4.6539e-01,  6.7843e-01,
#>           5.4638e-01, -5.2910e-01, -7.9840e-01,  1.1020e+00,  7.6241e-01],
#>         [-1.0718e+00, -2.4077e+00, -6.5038e-01,  2.7858e-01,  2.8280e+00,
#>           1.6169e-01, -1.8347e-01, -5.0920e-01,  2.6854e-01,  1.4414e+00],
#>         [-1.8737e+00,  4.4057e-01, -2.3574e-01, -6.9661e-01, -5.7052e-02,
#>          -8.6405e-01, -1.1220e+00,  2.9609e+00,  2.4298e-01,  1.1746e+00],
#>         [ 2.2905e-02, -1.3900e+00, -1.0668e+00,  1.8178e+00, -2.5003e-01,
#>           1.3210e+00,  6.9670e-01, -4.9706e-01,  1.5064e-02, -1.0484e-01],
#>         [-1.9512e-01, -2.0620e+00,  7.2796e-01, -1.2614e+00,  1.1771e+00,
#>          -4.4560e-01,  2.4300e+00, -2.5322e-01, -5.0160e-01,  3.4287e-01],
#>         [-1.7895e+00,  2.9237e+00,  5.5602e-01,  1.6005e-02, -6.8446e-01,
#>          -5.6196e-01, -3.0825e-01, -3.1230e-01,  5.3515e-01, -5.1871e-01],
#>         [ 1.2778e-01, -5.7787e-01, -2.3668e-01,  2.7850e+00, -1.2496e+00,
#>           7.2172e-01, -1.2078e+00, -4.5979e-01,  6.6433e-01, -6.4656e-01],
#>         [-1.1795e+00, -7.7226e-01,  5.4593e-01, -8.4331e-01,  3.5793e-01,
#>           1.4168e-02,  2.6910e+00, -1.3148e+00,  1.4298e-01,  2.4178e-01],
#>         [-7.0872e-01,  7.4130e-02,  1.4030e-01, -5.7927e-01,  5.9689e-01,
#>          -1.3790e-01, -2.6661e-01, -6.8908e-02,  6.6680e-01,  2.8735e-01],
#>         [-9.4736e-01, -3.8082e-01, -8.1021e-01,  2.5420e+00, -1.4745e+00,
#>           1.1623e+00, -9.1971e-01,  5.8198e-01,  4.8797e-01, -6.0779e-02],
#>         [-2.1313e+00,  1.6627e+00, -6.2109e-02,  1.8626e-01, -4.9098e-01,
#>           3.1607e-01,  3.0959e-01, -2.9732e-01,  8.1288e-01,  6.6196e-02],
#>         [-9.3417e-01, -5.4787e-01, -2.9066e-01, -1.3079e+00,  2.1613e+00,
#>          -5.5043e-01,  2.7881e-01, -3.1916e-02, -4.8963e-02,  1.4018e+00],
#>         [-1.1491e+00,  4.5298e-01, -2.4254e-01,  3.1110e-01, -1.0758e-01,
#>           4.3531e-01, -9.7760e-02,  7.9330e-02,  4.1125e-01,  2.2789e-01],
#>         [-1.8481e+00,  1.6300e+00, -5.0357e-01,  1.6786e-01, -1.3542e-01,
#>          -1.5521e-01, -1.1403e-02,  6.9099e-01,  2.5018e-01,  4.9468e-01],
#>         [ 7.5237e-01, -1.2569e+00,  6.4591e-01, -9.2931e-01, -3.4462e-01,
#>           4.9351e-01,  1.8210e+00, -1.1082e+00, -1.1965e-01, -3.3728e-01],
#>         [-1.2128e+00, -1.6614e+00,  6.5734e-02, -4.0805e-01,  1.6226e+00,
#>          -8.9240e-01, -3.6762e-01,  8.9544e-01,  4.9617e-02,  2.1655e+00]],
#>        grad_fn=<AddmmBackward>)
print(predicted)
#> tensor([8, 9, 0, 8, 2, 7, 4, 8, 6, 7, 8, 0, 8, 2, 9, 4, 7, 8, 4, 7, 8, 6, 4, 1,
#>         4, 2, 2, 4, 4, 7, 0, 1, 9, 2, 8, 7, 8, 2, 6, 0, 0, 6, 3, 8, 8, 9, 1, 4,
#>         0, 6, 1, 0, 0, 0, 0, 8, 1, 7, 7, 1, 4, 6, 0, 7, 0, 3, 6, 8, 7, 1, 3, 2,
#>         4, 4, 4, 2, 6, 4, 1, 7, 2, 6, 6, 0, 1, 7, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2,
#>         3, 4, 1, 6])
```

### Printing output size
This produces a 100x10 matrix because each iteration has a batch size of 100 and each prediction across the 10 classes, with the largest number indicating the likely number it is predicting. 


```r
# Iterate through test dataset
iter_test <- 0
iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
for (test_obj in iterate(iter_test_dataset)) {
    iter_test <- iter_test + 1
    # Load images to a Torch Variable
    images <- test_obj[[2]][[1]]
    labels <- test_obj[[2]][[2]]
    images <- images$view(-1L, 28L*28L)$requires_grad_()
    
    # Forward pass only to get logits/output
    outputs = model(images)
    
    if (iter_test == 1) {
        print('OUTPUTS')
        print(outputs$size())
    }
    # Get predictions from the maximum value
    .predicted = torch$max(outputs$data, 1L)
    predicted <- .predicted[1L]
}
#> [1] "OUTPUTS"
#> torch.Size([100, 10])
print(predicted$size())
#> torch.Size([100])
```

> The `predicted` and `output` tensors have the same number of 1D members. It is obvious because `predicted` is calculated from the `output` maximum values.

### Printing one output
This would be a 1x10 matrix where the largest number is what the model thinks the image is. Here we can see that in the tensor, position 7 has the largest number, indicating the model thinks the image is 7.

```
number 0: -0.4181
number 1: -1.0784
...
number 7: 2.9352
```


```r
# Iterate through test dataset
iter_test <- 0
iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
for (test_obj in iterate(iter_test_dataset)) {
    iter_test <- iter_test + 1
    # Load images to a Torch Variable
    images <- test_obj[[2]][[1]]
    labels <- test_obj[[2]][[2]]
    images <- images$view(-1L, 28L*28L)$requires_grad_()
    
    # Forward pass only to get logits/output
    outputs = model(images)
    
    if (iter_test == 1) {
        print('OUTPUTS')
        print(outputs[1])    # show first tensor of 100
        print(outputs[99])   # show last tensor of 100
    }
    # Get predictions from the maximum value for 100 tensors
    .predicted = torch$max(outputs$data, 1L)
    predicted <- .predicted[1L]
}
#> [1] "OUTPUTS"
#> Warning in `[.torch.Tensor`(outputs, 1): Incorrect number of dimensions
#> supplied. The number of supplied arguments, (not counting any NULL,
#> tf$newaxis or np$newaxis) must match thenumber of dimensions in the tensor,
#> unless an all_dims() was supplied (this will produce an error in the
#> future)
#> tensor([-0.4472, -1.2468, -0.2042, -0.2056,  0.0498, -0.4331, -1.0500,  2.8732,
#>         -0.1783,  0.8953], grad_fn=<SelectBackward>)
#> Warning in `[.torch.Tensor`(outputs, 99): Incorrect number of dimensions
#> supplied. The number of supplied arguments, (not counting any NULL,
#> tf$newaxis or np$newaxis) must match thenumber of dimensions in the tensor,
#> unless an all_dims() was supplied (this will produce an error in the
#> future)
#> tensor([ 0.7524, -1.2569,  0.6459, -0.9293, -0.3446,  0.4935,  1.8210, -1.1082,
#>         -0.1197, -0.3373], grad_fn=<SelectBackward>)
print(predicted)
#> tensor([8, 9, 0, 8, 2, 7, 4, 8, 6, 7, 8, 0, 8, 2, 9, 4, 7, 8, 4, 7, 8, 6, 4, 1,
#>         4, 2, 2, 4, 4, 7, 0, 1, 9, 2, 8, 7, 8, 2, 6, 0, 0, 6, 3, 8, 8, 9, 1, 4,
#>         0, 6, 1, 0, 0, 0, 0, 8, 1, 7, 7, 1, 4, 6, 0, 7, 0, 3, 6, 8, 7, 1, 3, 2,
#>         4, 4, 4, 2, 6, 4, 1, 7, 2, 6, 6, 0, 1, 7, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2,
#>         3, 4, 1, 6])
```

### Printing prediction output
Because our output is of size 100 (our batch size), our prediction size would also of the size 100.


```r
# Iterate through test dataset
iter_test <- 0
iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
for (test_obj in iterate(iter_test_dataset)) {
    iter_test <- iter_test + 1
    # Load images to a Torch Variable
    images <- test_obj[[2]][[1]]
    labels <- test_obj[[2]][[2]]
    images <- images$view(-1L, 28L*28L)$requires_grad_()
    
    # Forward pass only to get logits/output
    outputs = model(images)
    
    # Get predictions from the maximum value for a batch
    .predicted = torch$max(outputs$data, 1L)
    predicted <- .predicted[1L]
    
    if (iter_test == 1) {
        print('PREDICTION')
        print(predicted$size())
    }
}
#> [1] "PREDICTION"
#> torch.Size([100])
```

### Print prediction value
We are printing our prediction which as verified above, should be digit 7.


```r
# Iterate through test dataset
iter_test <- 0
iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
for (test_obj in iterate(iter_test_dataset)) {
    iter_test <- iter_test + 1
    # Load images to a Torch Variable
    images <- test_obj[[2]][[1]]
    labels <- test_obj[[2]][[2]]
    images <- images$view(-1L, 28L*28L)$requires_grad_()
    
    # Forward pass only to get logits/output
    outputs = model(images)
    
    # Get predictions from the maximum value
    .predicted = torch$max(outputs$data, 1L)
    predicted <- .predicted[1L]
    
    if (iter_test == 1) {
        print('PREDICTION')
        print(predicted[1L])
    }
}
#> [1] "PREDICTION"
#> tensor(7)
```

### Print prediction, label and label size
We are trying to show what we are predicting and the actual values. In this case, we're predicting the right value 7!


```r
# Iterate through test dataset
iter_test <- 0
iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
for (test_obj in iterate(iter_test_dataset)) {
    iter_test <- iter_test + 1
    # Load images to a Torch Variable
    images <- test_obj[[2]][[1]]
    labels <- test_obj[[2]][[2]]
    images <- images$view(-1L, 28L*28L)$requires_grad_()
    # Forward pass only to get logits/output
    outputs = model(images)
    # Get predictions from the maximum value
    .predicted = torch$max(outputs$data, 1L)
    predicted <- .predicted[1L]
    
    if (iter_test == 1) {
        print('PREDICTION')
        print(predicted[1])
        
        print('LABEL SIZE')
        print(labels$size())

        print('LABEL FOR IMAGE 0')
        print(labels[1]$item())  # extract the scalar part only
    }
}
#> [1] "PREDICTION"
#> tensor(7)
#> [1] "LABEL SIZE"
#> torch.Size([100])
#> [1] "LABEL FOR IMAGE 0"
#> [1] 7
```

### Print second prediction and ground truth
It should be the digit 2.


```r
# Iterate through test dataset
iter_test <- 0
iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
for (test_obj in iterate(iter_test_dataset)) {
    iter_test <- iter_test + 1
    # Load images to a Torch Variable
    images <- test_obj[[2]][[1]]
    labels <- test_obj[[2]][[2]]
    images <- images$view(-1L, 28L*28L)$requires_grad_()
    # Forward pass only to get logits/output
    outputs = model(images)
    # Get predictions from the maximum value
    .predicted = torch$max(outputs$data, 1L)
    predicted <- .predicted[1L]
    
    if (iter_test == 1) {
        print('PREDICTION')
        print(predicted[1])
        
        print('LABEL SIZE')
        print(labels$size())

        print('LABEL FOR IMAGE 0')
        print(labels[1]$item())
    }
}
#> [1] "PREDICTION"
#> tensor(7)
#> [1] "LABEL SIZE"
#> torch.Size([100])
#> [1] "LABEL FOR IMAGE 0"
#> [1] 7
```

### Print accuracy
Now we know what each object represents, we can understand how we arrived at our accuracy numbers.

One last thing to note is that `correct.item()` has this syntax is because correct is a PyTorch tensor and to get the value to compute with total which is an integer, we need to do this. 


```r
# Iterate through test dataset
iter_test <- 0
iter_test_dataset <- builtins$enumerate(test_loader) # convert to iterator
for (test_obj in iterate(iter_test_dataset)) {
    iter_test <- iter_test + 1
    # Load images to a Torch Variable
    images <- test_obj[[2]][[1]]
    labels <- test_obj[[2]][[2]]
    images <- images$view(-1L, 28L*28L)$requires_grad_()
    # Forward pass only to get logits/output
    outputs = model(images)
    # Get predictions from the maximum value
    .predicted = torch$max(outputs$data, 1L)
    predicted <- .predicted[1L]
    # Total number of labels
    total = total + labels$size(0L)
    # Total correct predictions
    correct = correct + sum((predicted$numpy() == labels$numpy()))
}
accuracy = 100 * correct / total
print(accuracy)
#> [1] 83.3
```

## Saving PyTorch model
This is how you save your model. 
Feel free to just change `save_model = TRUE` to save your model 


```r
save_model = TRUE
if (save_model) {
    # Saves only parameters
    torch$save(model$state_dict(), 'awesome_model.pkl')
}
```

