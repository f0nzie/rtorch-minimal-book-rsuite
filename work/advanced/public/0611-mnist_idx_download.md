
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
#> <generator object Module.parameters at 0x7f2d73b0d150>
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
#> Iteration:   500. Loss: 1.868431. Accuracy:    67.11 
#> Iteration:  1000. Loss: 1.515333. Accuracy:    76.19 
#> Iteration:  1500. Loss: 1.353359. Accuracy:    79.17 
#> Iteration:  2000. Loss: 1.088589. Accuracy:    80.87 
#> Iteration:  2500. Loss: 1.154811. Accuracy:    82.05 
#> Iteration:  3000. Loss: 0.983550. Accuracy:    82.80
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
#> tensor([[-4.4140e-01, -1.1377e+00, -4.8739e-01, -2.1394e-01, -1.7316e-02,
#>          -3.5983e-01, -1.0735e+00,  2.6029e+00, -2.8652e-01,  7.7327e-01],
#>         [ 4.7324e-01,  1.0631e-01,  1.4385e+00,  1.2873e+00, -1.8725e+00,
#>           6.3831e-01,  1.2380e+00, -1.7969e+00,  4.7130e-01, -1.5070e+00],
#>         [-8.8870e-01,  2.2625e+00,  2.3038e-01,  1.4436e-01, -5.8951e-01,
#>          -2.4195e-01, -7.2440e-02, -1.5871e-01,  1.9768e-01, -4.3548e-01],
#>         [ 2.8079e+00, -2.2973e+00, -1.4636e-01, -1.1471e-01, -9.2921e-01,
#>           5.3865e-01,  1.2454e+00,  1.1918e-01, -6.9602e-01, -2.0122e-01],
#>         [-2.1243e-01, -1.9088e+00,  4.1438e-01, -6.5429e-01,  1.7735e+00,
#>          -4.3085e-01, -2.5054e-02,  5.0409e-01, -1.0392e-02,  7.8818e-01],
#>         [-1.2979e+00,  2.7598e+00,  1.7838e-01,  3.2013e-01, -6.1278e-01,
#>          -3.7355e-01, -6.1336e-01, -7.4954e-02,  4.2683e-01, -3.3123e-01],
#>         [-1.1819e+00, -1.0832e+00, -6.2141e-01,  2.1052e-01,  1.5254e+00,
#>           3.9585e-01, -8.8334e-01,  7.5761e-01,  5.4634e-01,  8.2142e-01],
#>         [-1.4164e+00, -3.2331e-01, -5.3094e-01, -1.0548e-01,  8.5932e-01,
#>           2.8414e-01,  3.7002e-01, -1.9190e-02,  2.5902e-01,  1.4571e+00],
#>         [ 1.4808e-01, -3.3844e-01,  8.0639e-01, -1.2379e+00,  6.1662e-01,
#>           1.1116e-01,  9.1921e-01, -7.4276e-01, -1.4782e-02,  1.6027e-01],
#>         [-4.4877e-01, -8.6476e-01, -1.2187e+00, -1.0574e+00,  1.0151e+00,
#>          -4.0089e-01, -5.3822e-01,  1.4898e+00,  1.2080e-01,  1.5985e+00],
#>         [ 3.2609e+00, -1.7940e+00,  4.0050e-01,  9.8763e-01, -7.3191e-01,
#>           1.1296e+00, -7.0141e-01, -1.5531e+00,  5.2794e-01, -1.8464e+00],
#>         [ 9.8359e-01, -4.4643e-01,  6.2426e-01,  9.2992e-02,  4.3961e-01,
#>          -4.6203e-01,  5.8461e-01, -8.1946e-01,  2.8264e-01, -6.4712e-01],
#>         [-8.4485e-01, -1.4076e+00, -8.0521e-01, -1.1333e+00,  1.6293e+00,
#>          -2.0828e-01, -2.0268e-01,  1.1923e+00,  2.5666e-01,  2.2458e+00],
#>         [ 3.0669e+00, -2.4332e+00, -1.9311e-01, -1.9413e-01, -3.4288e-01,
#>           8.1166e-01, -1.7193e-01, -8.2927e-01,  4.6970e-01,  2.7220e-01],
#>         [-1.3738e+00,  2.8071e+00, -6.0439e-02,  4.6022e-01, -1.0549e+00,
#>          -9.8678e-02,  1.3134e-01, -1.6560e-01,  3.0662e-01, -3.2230e-01],
#>         [ 4.9559e-01, -8.4138e-01,  6.8178e-02,  1.2115e+00, -2.8759e-01,
#>           9.3907e-01, -4.0092e-01, -7.3150e-01,  5.3912e-01, -7.8031e-01],
#>         [-4.2756e-01, -1.8421e+00,  1.3053e-01, -5.5626e-01,  1.4984e+00,
#>          -5.6205e-01, -2.2112e-01,  9.8104e-01,  5.2377e-02,  1.5797e+00],
#>         [ 6.7669e-03, -1.3608e+00, -7.6507e-01,  4.7372e-01, -2.4316e-01,
#>          -1.1835e-01, -6.7901e-01,  2.4755e+00, -4.7864e-01,  4.9072e-01],
#>         [-6.2523e-01, -2.7532e-01,  1.4597e-01,  1.3773e+00, -6.3374e-01,
#>           9.0506e-01,  1.0225e+00, -8.1359e-01,  2.4012e-01, -4.2973e-01],
#>         [-9.8784e-01, -1.4354e+00, -4.2955e-01, -8.6710e-02,  1.9530e+00,
#>           6.3454e-02, -1.7421e-01,  1.9932e-01,  4.6731e-02,  1.4883e+00],
#>         [-7.0393e-01, -2.5698e-01, -1.3459e+00,  1.7735e-01,  4.9735e-01,
#>           1.2960e-01, -1.3370e+00,  1.6259e+00,  3.7332e-01,  1.3775e+00],
#>         [-7.3933e-01, -1.3902e+00, -6.7426e-03,  2.6669e-01,  4.2416e-01,
#>           5.8904e-01,  2.1874e+00, -1.3912e+00,  3.1417e-01,  2.3564e-01],
#>         [-4.5361e-01,  1.2611e-01,  3.4747e-01, -3.9537e-01,  8.9021e-01,
#>          -9.4419e-01,  1.2301e+00,  2.0668e-02, -2.7363e-01, -2.7609e-01],
#>         [ 2.9692e-02, -8.6171e-01, -4.5364e-01,  4.0124e-01, -1.7968e-03,
#>           1.4392e+00,  3.5053e-01, -8.1117e-01,  9.7654e-01,  3.0010e-01],
#>         [-6.8996e-01, -1.1288e+00,  1.3267e-01, -1.9080e-01,  1.4716e+00,
#>          -2.0411e-01, -2.1137e-01,  5.4029e-01, -2.0452e-01,  9.3689e-01],
#>         [ 4.2808e+00, -2.7909e+00,  5.2460e-01, -1.2982e+00, -2.8380e-01,
#>           1.2518e+00,  8.1313e-01, -9.2210e-01, -1.9554e-01, -1.3746e+00],
#>         [-3.5220e-01, -1.1680e+00, -3.7162e-01,  2.6507e-01,  9.6316e-02,
#>          -9.0560e-02, -5.7929e-01,  1.6551e+00, -3.3500e-01,  1.2070e+00],
#>         [-6.7561e-01, -2.2356e+00, -2.1182e-01, -7.7566e-01,  2.4611e+00,
#>          -3.4163e-02,  1.4421e-01, -4.1675e-02,  9.0148e-02,  1.5197e+00],
#>         [ 2.7103e+00, -2.3570e+00,  1.8045e-01,  1.0699e+00, -1.1652e+00,
#>           7.5705e-01, -9.6590e-02, -7.2810e-01,  5.8475e-01, -6.7024e-01],
#>         [-1.1404e+00,  1.3969e+00, -3.2811e-01,  3.1693e-01, -3.7244e-01,
#>           2.7442e-01,  1.7315e-01, -1.5959e-01,  3.5467e-01, -1.4939e-01],
#>         [-7.8069e-01, -1.0423e-01, -9.4658e-01,  2.2308e+00, -1.1385e+00,
#>           1.0306e+00, -3.0668e-01,  6.8090e-01,  3.8931e-01, -1.7012e-01],
#>         [-1.1840e+00,  1.1664e+00, -2.0020e-01,  2.7470e-01, -3.7369e-01,
#>           1.2383e-01, -9.0252e-02,  2.4221e-01,  1.8673e-01,  2.0008e-02],
#>         [-1.0032e+00, -7.0574e-01, -8.2568e-01,  2.4624e+00, -2.7302e-01,
#>           1.3707e+00, -3.6739e-01, -8.9338e-01,  6.9138e-01, -6.4864e-02],
#>         [ 1.8303e+00, -1.5592e+00,  5.7662e-01, -1.7426e+00,  7.8629e-01,
#>           1.6069e-01,  1.4888e+00, -9.6620e-01,  6.2431e-02, -9.1792e-03],
#>         [-9.5319e-01, -3.6774e-01,  9.1068e-01, -3.6694e-01, -2.3987e-03,
#>          -3.8399e-01, -1.5005e+00,  1.8105e+00,  7.7263e-01,  3.5485e-01],
#>         [ 2.9729e-01, -1.3747e+00,  2.5310e+00,  4.8956e-03, -7.2437e-01,
#>           2.1619e-01,  2.1320e-01, -3.4970e-01,  2.7646e-01, -2.0226e+00],
#>         [-6.9694e-01, -1.4696e+00,  2.1671e-01, -2.4897e-01, -2.6074e-02,
#>          -2.5968e-01, -5.4975e-01,  2.2435e+00, -1.6442e-01,  8.9847e-01],
#>         [-1.4751e+00,  2.1489e+00, -3.7231e-01,  7.5580e-02, -5.3102e-01,
#>           1.4797e-01,  1.0389e-01, -4.2931e-02,  3.7249e-01,  4.4789e-03],
#>         [ 4.5079e-01,  5.9094e-01,  9.6301e-01,  1.1970e+00, -1.7786e+00,
#>           2.6764e-01,  6.8134e-01, -1.0380e+00,  8.1396e-01, -1.3053e+00],
#>         [-1.2887e+00,  3.0655e+00, -1.7339e-01,  5.4016e-01, -1.2735e+00,
#>           5.2410e-03,  1.5545e-01, -4.6484e-01,  7.0311e-01, -1.8979e-01],
#>         [-7.1235e-01,  1.5216e+00, -1.2919e-01,  6.3709e-02, -4.6907e-01,
#>          -9.7585e-02,  5.9337e-03, -2.8869e-02,  2.8825e-02, -2.1682e-01],
#>         [-7.9474e-01, -7.2051e-01, -2.9145e-02, -5.0352e-01, -4.5921e-02,
#>          -3.9645e-01, -4.4087e-01,  2.1888e+00, -2.0438e-01,  1.3294e+00],
#>         [-2.0022e+00, -3.0243e-01, -1.4978e-01, -4.9997e-01,  2.2832e+00,
#>          -6.9153e-01, -8.7564e-01,  6.7739e-01,  3.8978e-01,  1.5857e+00],
#>         [-1.4078e-01,  1.0079e+00,  1.4149e+00, -3.1387e-02, -3.7300e-02,
#>          -3.7499e-01,  1.9420e-01, -1.1263e+00,  3.7233e-01, -3.8530e-01],
#>         [-1.0250e+00,  6.9793e-02, -8.8931e-02,  1.4547e+00, -6.8809e-01,
#>           5.2916e-01,  3.6399e-01, -4.3179e-02,  9.4762e-02, -2.8480e-01],
#>         [ 1.4282e-02, -1.1136e+00, -6.3311e-01,  1.5326e+00, -3.9917e-01,
#>           1.4791e+00, -1.1071e-01, -1.2850e+00,  1.0327e+00, -3.2173e-01],
#>         [-1.8163e+00,  2.7865e-01,  1.4349e-01,  7.2162e-01, -3.7825e-02,
#>           4.3462e-01,  1.3197e-01,  2.1985e-01,  3.9200e-01,  1.9086e-01],
#>         [-8.4963e-01, -1.5366e-01,  1.6077e+00, -3.5872e-01,  4.4691e-01,
#>          -7.9414e-01,  6.4319e-01, -4.5973e-01, -1.3198e-02,  8.6972e-03],
#>         [-1.1928e+00, -2.5667e+00, -1.1624e+00, -5.7610e-02,  2.8632e+00,
#>           4.1446e-01, -7.8069e-01,  4.7991e-01,  6.6292e-01,  2.1669e+00],
#>         [-3.5097e-01, -1.9589e+00,  5.2370e-01, -6.8944e-01,  2.4526e+00,
#>          -1.0303e+00,  2.6327e-01,  2.9367e-01, -2.9351e-01,  9.3290e-01],
#>         [-6.5777e-02, -8.2858e-01,  1.2045e-01,  3.5643e-01, -3.4589e-01,
#>           6.7464e-01,  2.1103e+00, -8.3783e-01, -1.0746e-01, -4.1369e-01],
#>         [ 5.6197e-02, -8.4131e-01, -9.4903e-02,  2.0142e+00, -5.4962e-01,
#>           7.3387e-01,  3.3788e-01, -3.6045e-01, -1.3525e-01, -3.8195e-02],
#>         [ 4.5485e-01, -1.1575e+00, -1.3941e+00,  1.5094e-01,  8.7438e-01,
#>           1.2743e+00, -3.6548e-02, -7.5872e-02,  5.5531e-02,  7.9110e-01],
#>         [ 3.9007e-01, -8.1184e-01, -1.1995e-01,  1.1958e+00,  2.4778e-03,
#>           7.3665e-01, -1.6001e-01, -7.2304e-01,  4.9545e-01, -3.7770e-01],
#>         [ 1.6367e-01, -4.5132e-01,  1.3647e+00, -2.7148e-01,  4.8319e-01,
#>          -5.0809e-01,  9.8298e-01, -3.6708e-01,  4.7603e-02, -7.9513e-01],
#>         [ 1.3936e+00, -2.1594e+00, -6.5466e-01,  6.5891e-01, -6.8545e-01,
#>           9.6739e-01,  3.4473e-01, -1.1109e+00,  1.1495e+00, -6.6569e-03],
#>         [-1.9194e-01, -2.7286e+00, -6.3881e-02, -4.9257e-01,  2.7539e+00,
#>           3.0878e-04,  1.1398e-01, -1.5829e-01,  1.7726e-04,  9.0614e-01],
#>         [-8.6049e-01,  2.3499e+00,  3.8955e-02,  4.1840e-01, -7.9283e-01,
#>          -3.0456e-01, -3.1743e-01, -4.9671e-02,  3.3337e-01, -2.4579e-01],
#>         [-4.1817e-01, -1.8570e+00, -9.5969e-01, -6.4785e-01,  1.8247e+00,
#>          -3.2061e-01, -5.9911e-02,  9.5460e-01, -3.4781e-01,  2.1997e+00],
#>         [ 9.2170e-02,  3.2568e-01, -6.3023e-01, -2.5725e-01,  2.9330e-01,
#>           2.3740e-01, -1.6686e-01,  3.2227e-01,  3.4870e-01, -1.1806e-01],
#>         [-4.2606e-01, -1.6728e+00, -5.4970e-01,  7.5001e-01,  2.9911e-03,
#>           2.8410e-01, -2.9644e-01,  2.0625e+00, -4.7992e-01,  6.5383e-01],
#>         [ 4.4258e-01, -8.9746e-01,  1.2949e+00, -1.5720e+00,  9.0857e-03,
#>          -1.0465e-01,  6.9170e-01, -6.8186e-01,  1.1549e+00,  3.0782e-01],
#>         [-8.6044e-01, -6.9274e-01,  4.8345e-02, -4.0279e-01,  7.0805e-01,
#>           1.0653e-01, -6.9607e-02,  1.6207e-01,  5.5467e-01,  9.3291e-01],
#>         [-7.0451e-01,  2.6125e-02,  1.2664e+00,  7.1237e-01, -1.3444e-01,
#>          -3.3386e-01, -4.5315e-01, -5.8216e-01,  4.2480e-01,  2.6113e-01],
#>         [-1.0845e+00, -8.3157e-01,  5.0279e-01, -5.1181e-01,  8.2834e-01,
#>          -4.5235e-01, -6.2129e-01,  1.5252e+00,  1.8812e-01,  6.7038e-01],
#>         [-1.2901e+00, -4.5933e-01, -4.2986e-01,  6.1075e-01,  6.0412e-01,
#>           5.2704e-01, -1.2437e-02, -1.2406e-01,  6.6960e-01,  9.0447e-01],
#>         [ 4.6331e-01, -4.4293e-01,  7.6583e-01, -1.3096e-01,  5.0651e-01,
#>          -7.0491e-01,  6.4419e-01,  3.1496e-01, -5.3405e-01, -7.3465e-01],
#>         [-7.4146e-01, -1.3001e+00,  5.5543e-01, -8.0594e-01,  2.2054e+00,
#>          -6.6841e-01, -1.6948e-01,  4.4136e-01,  1.0831e-01,  7.3052e-01],
#>         [-1.3624e+00, -1.3605e-01, -2.3461e-01,  2.8401e+00, -3.8335e-01,
#>           8.1241e-01, -8.4256e-01, -6.4804e-01,  1.1170e+00, -7.6314e-02],
#>         [ 2.8003e+00, -1.5381e+00,  7.9431e-01, -6.7078e-01, -1.3093e+00,
#>           5.7365e-01,  4.7153e-01,  7.9551e-02, -2.8470e-01, -5.1648e-01],
#>         [ 2.0264e-01, -1.1904e+00, -8.0854e-01, -9.5482e-02, -3.0399e-01,
#>          -1.7275e-01, -6.9299e-01,  2.6143e+00, -4.3995e-01,  6.0178e-01],
#>         [ 4.4881e+00, -2.9172e+00,  9.2259e-01,  2.6830e-01, -8.2826e-01,
#>           1.1383e+00,  9.7329e-02, -1.6387e+00,  2.6445e-01, -1.6806e+00],
#>         [ 1.6201e+00, -1.1930e+00,  1.5600e+00,  1.2528e+00, -9.8180e-01,
#>           3.9447e-01,  1.8346e-01, -1.0228e+00,  3.2264e-01, -1.9944e+00],
#>         [-1.3417e+00,  7.4961e-01,  4.2059e-01,  6.6279e-02, -6.0815e-01,
#>          -2.6390e-01, -8.7575e-01,  9.2690e-01,  1.3315e+00,  5.4316e-01],
#>         [-1.3807e+00,  2.3746e+00, -4.1210e-01,  6.6408e-02, -8.8007e-01,
#>           5.2352e-02,  2.1954e-01, -9.4525e-02,  5.9019e-01, -1.1594e-01],
#>         [-1.8909e+00,  5.8913e-01, -4.7865e-01, -5.6890e-01,  9.9365e-01,
#>          -4.4993e-01, -7.3636e-01,  1.8792e+00,  1.3028e-01,  9.1642e-01],
#>         [-3.7366e-01,  1.9046e-01, -2.5045e-01,  2.4083e+00, -8.6841e-01,
#>           1.0653e+00, -4.6659e-01, -9.0421e-01,  6.0239e-01, -6.3714e-01],
#>         [-9.9534e-01, -3.2040e-01,  8.7107e-01, -5.6641e-01, -2.2977e-01,
#>          -6.6326e-01,  1.4039e-01,  1.6457e+00, -2.7698e-01,  2.8842e-01],
#>         [-1.5247e+00,  1.1076e+00, -9.6303e-01,  4.4011e-01, -1.1498e-01,
#>           5.9345e-02, -6.1593e-01,  3.3809e-01,  9.4765e-01,  6.2457e-01],
#>         [-5.4832e-01, -7.4808e-01, -4.0993e-01, -9.6398e-01, -1.7236e-02,
#>          -3.5352e-01, -1.3842e+00,  3.2990e+00,  4.3496e-01,  7.7785e-01],
#>         [-7.0883e-01, -1.6701e+00, -9.4763e-01, -2.4894e-01,  7.4348e-01,
#>           4.7424e-01, -3.4954e-01,  1.6933e+00, -3.2268e-01,  1.6493e+00],
#>         [-4.0033e-02, -1.7949e+00,  7.5340e-01, -7.3914e-01,  1.8716e-01,
#>           2.2872e-01,  2.5163e+00, -4.0696e-01, -3.7693e-01,  1.7780e-01],
#>         [-6.6123e-01, -9.0361e-01,  3.9220e+00, -3.8973e-01,  1.6901e-01,
#>          -1.0962e+00,  6.6343e-01, -1.0166e+00,  5.3942e-01, -1.3306e+00],
#>         [-7.4825e-01, -1.5783e+00, -5.2694e-01,  5.4008e-02,  7.2422e-01,
#>          -5.9870e-02, -8.6626e-01,  2.0369e+00, -2.9308e-01,  1.4136e+00],
#>         [-7.8115e-01,  1.7886e-01, -2.0192e-01, -4.6541e-01,  8.0600e-01,
#>           7.2173e-01, -3.2362e-01, -4.9461e-01,  1.1491e+00,  8.8406e-01],
#>         [-1.2189e+00, -2.2974e+00, -8.1705e-01,  3.5329e-01,  3.0015e+00,
#>           3.2803e-01, -3.7379e-01, -1.9106e-01,  3.9180e-01,  1.4544e+00],
#>         [-1.8946e+00,  6.0085e-01, -1.7806e-01, -6.2187e-01, -1.3003e-01,
#>          -7.9969e-01, -1.1188e+00,  2.9647e+00,  3.1092e-01,  8.5111e-01],
#>         [ 2.8342e-03, -1.3234e+00, -1.2654e+00,  1.8597e+00, -2.2351e-01,
#>           1.2507e+00,  6.3885e-01, -3.2966e-01,  1.2025e-01, -1.0005e-02],
#>         [-1.8925e-01, -2.2231e+00,  8.2887e-01, -1.2402e+00,  1.1085e+00,
#>          -6.0109e-01,  2.4343e+00, -9.4018e-02, -5.2147e-01,  2.8355e-01],
#>         [-1.5839e+00,  2.9318e+00,  5.8698e-01,  1.9129e-01, -6.1950e-01,
#>          -4.0846e-01, -2.2253e-01, -3.3789e-01,  4.6051e-01, -5.8730e-01],
#>         [ 3.1438e-01, -6.4303e-01, -5.4869e-02,  2.7954e+00, -9.9945e-01,
#>           9.3163e-01, -1.4260e+00, -5.7302e-01,  8.3677e-01, -6.1200e-01],
#>         [-1.1161e+00, -4.6436e-01,  4.0792e-01, -7.4245e-01,  4.1173e-01,
#>          -5.2051e-02,  2.8810e+00, -1.0619e+00,  2.0089e-01,  2.1789e-01],
#>         [-8.9775e-01,  1.2289e-01,  9.0853e-02, -5.7722e-01,  5.6920e-01,
#>          -2.7595e-02, -3.0293e-01, -1.2959e-02,  8.6259e-01,  3.7948e-01],
#>         [-7.8834e-01, -4.7971e-01, -6.8238e-01,  2.6277e+00, -1.3917e+00,
#>           1.0669e+00, -9.4160e-01,  8.5969e-01,  7.7495e-01, -6.1954e-02],
#>         [-2.2363e+00,  1.7754e+00,  1.4806e-01,  2.0250e-01, -3.2775e-01,
#>           2.5095e-01,  5.1321e-01, -9.8250e-02,  8.5426e-01, -6.6636e-03],
#>         [-9.8605e-01, -2.8013e-01, -1.3478e-01, -1.3560e+00,  2.2511e+00,
#>          -3.8334e-01,  5.9066e-01,  7.1568e-02, -1.0325e-01,  1.3802e+00],
#>         [-1.2965e+00,  5.6988e-01, -2.5008e-01,  4.5906e-01, -1.4842e-01,
#>           4.2227e-01, -1.2463e-01,  2.8340e-01,  4.6975e-01,  1.0592e-01],
#>         [-1.9350e+00,  1.4343e+00, -4.6094e-01,  6.4645e-02, -1.8978e-01,
#>           7.4429e-03,  7.5194e-02,  1.0580e+00,  4.9151e-01,  6.0092e-01],
#>         [ 5.8134e-01, -8.5205e-01,  7.7849e-01, -9.5335e-01, -3.0011e-01,
#>           5.8798e-01,  2.0786e+00, -1.1024e+00,  8.7965e-03, -5.1300e-01],
#>         [-1.1969e+00, -1.6240e+00,  3.9591e-01, -4.7555e-01,  1.6252e+00,
#>          -1.0191e+00, -1.7782e-01,  1.0649e+00,  1.3597e-01,  2.1946e+00]],
#>        grad_fn=<AddmmBackward>)
print(predicted)
#> tensor([8, 9, 0, 8, 4, 7, 4, 2, 6, 7, 8, 0, 8, 2, 8, 4, 7, 8, 9, 7, 8, 6, 4, 1,
#>         4, 7, 2, 4, 4, 7, 0, 1, 9, 2, 8, 7, 8, 2, 6, 0, 6, 6, 3, 8, 8, 4, 1, 4,
#>         0, 6, 1, 0, 0, 0, 0, 8, 1, 7, 7, 1, 4, 6, 0, 7, 0, 3, 6, 8, 7, 1, 3, 2,
#>         4, 4, 4, 2, 6, 4, 1, 7, 0, 2, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2,
#>         3, 4, 8, 6])
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
#> tensor([-0.4414, -1.1377, -0.4874, -0.2139, -0.0173, -0.3598, -1.0735,  2.6029,
#>         -0.2865,  0.7733], grad_fn=<SelectBackward>)
#> Warning in `[.torch.Tensor`(outputs, 99): Incorrect number of dimensions
#> supplied. The number of supplied arguments, (not counting any NULL,
#> tf$newaxis or np$newaxis) must match thenumber of dimensions in the tensor,
#> unless an all_dims() was supplied (this will produce an error in the
#> future)
#> tensor([ 0.5813, -0.8520,  0.7785, -0.9533, -0.3001,  0.5880,  2.0786, -1.1024,
#>          0.0088, -0.5130], grad_fn=<SelectBackward>)
print(predicted)
#> tensor([8, 9, 0, 8, 4, 7, 4, 2, 6, 7, 8, 0, 8, 2, 8, 4, 7, 8, 9, 7, 8, 6, 4, 1,
#>         4, 7, 2, 4, 4, 7, 0, 1, 9, 2, 8, 7, 8, 2, 6, 0, 6, 6, 3, 8, 8, 4, 1, 4,
#>         0, 6, 1, 0, 0, 0, 0, 8, 1, 7, 7, 1, 4, 6, 0, 7, 0, 3, 6, 8, 7, 1, 3, 2,
#>         4, 4, 4, 2, 6, 4, 1, 7, 0, 2, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2,
#>         3, 4, 8, 6])
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
#> [1] 82.8
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

