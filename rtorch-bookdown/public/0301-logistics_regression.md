
# (PART) Logistic Regression {-}

# Example 1: MNIST handwritten digits {#mnistdigits}

Source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/logistic_regression/main.py



```r
library(rTorch)

nn          <- torch$nn
transforms  <- torchvision$transforms

torch$set_default_dtype(torch$float)
```


## Hyperparameters

```r
# Hyper-parameters 
input_size    <- 784L
num_classes   <- 10L
num_epochs    <- 5L
batch_size    <- 100L
learning_rate <- 0.001
```

## Read datasets


```r
# MNIST dataset (images and labels)
# IDX format
local_folder <- './datasets/raw_data'
train_dataset = torchvision$datasets$MNIST(root=local_folder, 
                                           train=TRUE, 
                                           transform=transforms$ToTensor(),
                                           download=TRUE)

test_dataset = torchvision$datasets$MNIST(root=local_folder, 
                                          train=FALSE, 
                                          transform=transforms$ToTensor())

# Data loader (input pipeline). Make the datasets iteratble
train_loader = torch$utils$data$DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=TRUE)

test_loader = torch$utils$data$DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=FALSE)
```


```r
class(train_loader)
#> [1] "torch.utils.data.dataloader.DataLoader"
#> [2] "python.builtin.object"
length(train_loader)
#> [1] 2
```

## Define the model


```r
# Logistic regression model
model = nn$Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn$CrossEntropyLoss()  
optimizer = torch$optim$SGD(model$parameters(), lr=learning_rate)  
print(model)
#> Linear(in_features=784, out_features=10, bias=True)
```


## Training


```r
# Train the model
iter_train_loader <- iterate(train_loader)
total_step <-length(iter_train_loader)
```


```r
for (epoch in 1:num_epochs) {
    i <-  0
    for (obj in iter_train_loader) {
        
        images <- obj[[1]]   # tensor torch.Size([64, 3, 28, 28])
        labels <- obj[[2]]   # tensor torch.Size([64]), labels from 0 to 9
        # cat(i, "\t"); print(images$shape)

        # Reshape images to (batch_size, input_size)
        images <- images$reshape(-1L, 28L*28L)
        # images <- torch$as_tensor(images$reshape(-1L, 28L*28L), dtype=torch$double)

        # Forward pass
        outputs <- model(images)
        loss <- criterion(outputs, labels)

        # Backward and optimize
        optimizer$zero_grad()
        loss$backward()
        optimizer$step()

        if ((i+1) %% 100 == 0) {
            cat(sprintf('Epoch [%d/%d], Step [%d/%d], Loss: %f \n',
                epoch+1, num_epochs, i+1, total_step, loss$item()))
        }
        i <-  i + 1
    }
}  
#> Epoch [2/5], Step [100/600], Loss: 2.207937 
#> Epoch [2/5], Step [200/600], Loss: 2.129730 
#> Epoch [2/5], Step [300/600], Loss: 2.020010 
#> Epoch [2/5], Step [400/600], Loss: 1.921782 
#> Epoch [2/5], Step [500/600], Loss: 1.842267 
#> Epoch [2/5], Step [600/600], Loss: 1.849545 
#> Epoch [3/5], Step [100/600], Loss: 1.752690 
#> Epoch [3/5], Step [200/600], Loss: 1.671137 
#> Epoch [3/5], Step [300/600], Loss: 1.618696 
#> Epoch [3/5], Step [400/600], Loss: 1.548630 
#> Epoch [3/5], Step [500/600], Loss: 1.447902 
#> Epoch [3/5], Step [600/600], Loss: 1.540873 
#> Epoch [4/5], Step [100/600], Loss: 1.462256 
#> Epoch [4/5], Step [200/600], Loss: 1.385923 
#> Epoch [4/5], Step [300/600], Loss: 1.361354 
#> Epoch [4/5], Step [400/600], Loss: 1.314525 
#> Epoch [4/5], Step [500/600], Loss: 1.196100 
#> Epoch [4/5], Step [600/600], Loss: 1.338717 
#> Epoch [5/5], Step [100/600], Loss: 1.272773 
#> Epoch [5/5], Step [200/600], Loss: 1.201425 
#> Epoch [5/5], Step [300/600], Loss: 1.194291 
#> Epoch [5/5], Step [400/600], Loss: 1.157681 
#> Epoch [5/5], Step [500/600], Loss: 1.029743 
#> Epoch [5/5], Step [600/600], Loss: 1.199558 
#> Epoch [6/5], Step [100/600], Loss: 1.144258 
#> Epoch [6/5], Step [200/600], Loss: 1.076404 
#> Epoch [6/5], Step [300/600], Loss: 1.080425 
#> Epoch [6/5], Step [400/600], Loss: 1.046417 
#> Epoch [6/5], Step [500/600], Loss: 0.914617 
#> Epoch [6/5], Step [600/600], Loss: 1.098780
```


## Prediction


```r
# Adjust weights and reset gradients
iter_test_loader <- iterate(test_loader)

with(torch$no_grad(), {
    correct <-  0
    total <-  0
    for (obj in iter_test_loader) {
        images <- obj[[1]]   # tensor torch.Size([64, 3, 28, 28])
        labels <- obj[[2]]   # tensor torch.Size([64]), labels from 0 to 9
        images = images$reshape(-1L, 28L*28L)
        # images <- torch$as_tensor(images$reshape(-1L, 28L*28L), dtype=torch$double)
        outputs = model(images)
        .predicted = torch$max(outputs$data, 1L)
        predicted <- .predicted[1L]
        total = total + labels$size(0L)
        correct = correct + sum((predicted$numpy() == labels$numpy()))
    }
    cat(sprintf('Accuracy of the model on the 10000 test images: %f %%', (100 * correct / total)))
  
})
#> Accuracy of the model on the 10000 test images: 83.360000 %
```

## Save the model


```r
# Save the model checkpoint
torch$save(model$state_dict(), 'model.ckpt')
```
