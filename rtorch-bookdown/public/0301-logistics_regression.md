
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
local_folder <- '../datasets/raw_data'
train_dataset = torchvision$datasets$MNIST(root=local_folder, 
                                           train=TRUE, 
                                           transform=transforms$ToTensor(),
                                           download=TRUE)

test_dataset = torchvision$datasets$MNIST(root=local_folder, 
                                          train=FALSE, 
                                          transform=transforms$ToTensor())

# Data loader (input pipeline)
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
#> Epoch [2/5], Step [100/600], Loss: 2.228390 
#> Epoch [2/5], Step [200/600], Loss: 2.116164 
#> Epoch [2/5], Step [300/600], Loss: 2.019095 
#> Epoch [2/5], Step [400/600], Loss: 1.994817 
#> Epoch [2/5], Step [500/600], Loss: 1.876899 
#> Epoch [2/5], Step [600/600], Loss: 1.840720 
#> Epoch [3/5], Step [100/600], Loss: 1.690704 
#> Epoch [3/5], Step [200/600], Loss: 1.681273 
#> Epoch [3/5], Step [300/600], Loss: 1.619535 
#> Epoch [3/5], Step [400/600], Loss: 1.650310 
#> Epoch [3/5], Step [500/600], Loss: 1.541979 
#> Epoch [3/5], Step [600/600], Loss: 1.520510 
#> Epoch [4/5], Step [100/600], Loss: 1.362663 
#> Epoch [4/5], Step [200/600], Loss: 1.406761 
#> Epoch [4/5], Step [300/600], Loss: 1.364937 
#> Epoch [4/5], Step [400/600], Loss: 1.420470 
#> Epoch [4/5], Step [500/600], Loss: 1.323673 
#> Epoch [4/5], Step [600/600], Loss: 1.311175 
#> Epoch [5/5], Step [100/600], Loss: 1.151503 
#> Epoch [5/5], Step [200/600], Loss: 1.225978 
#> Epoch [5/5], Step [300/600], Loss: 1.196704 
#> Epoch [5/5], Step [400/600], Loss: 1.259773 
#> Epoch [5/5], Step [500/600], Loss: 1.173820 
#> Epoch [5/5], Step [600/600], Loss: 1.167180 
#> Epoch [6/5], Step [100/600], Loss: 1.008452 
#> Epoch [6/5], Step [200/600], Loss: 1.100466 
#> Epoch [6/5], Step [300/600], Loss: 1.079904 
#> Epoch [6/5], Step [400/600], Loss: 1.142588 
#> Epoch [6/5], Step [500/600], Loss: 1.065565 
#> Epoch [6/5], Step [600/600], Loss: 1.062986
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
#> Accuracy of the model on the 10000 test images: 82.830000 %
```

## Save the model


```r
# Save the model checkpoint
torch$save(model$state_dict(), 'model.ckpt')
```
