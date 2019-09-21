
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
#> Epoch [2/5], Step [100/600], Loss: 2.199810 
#> Epoch [2/5], Step [200/600], Loss: 2.105293 
#> Epoch [2/5], Step [300/600], Loss: 2.047850 
#> Epoch [2/5], Step [400/600], Loss: 1.932216 
#> Epoch [2/5], Step [500/600], Loss: 1.887103 
#> Epoch [2/5], Step [600/600], Loss: 1.813347 
#> Epoch [3/5], Step [100/600], Loss: 1.718015 
#> Epoch [3/5], Step [200/600], Loss: 1.671756 
#> Epoch [3/5], Step [300/600], Loss: 1.634644 
#> Epoch [3/5], Step [400/600], Loss: 1.536070 
#> Epoch [3/5], Step [500/600], Loss: 1.557228 
#> Epoch [3/5], Step [600/600], Loss: 1.497406 
#> Epoch [4/5], Step [100/600], Loss: 1.404598 
#> Epoch [4/5], Step [200/600], Loss: 1.391037 
#> Epoch [4/5], Step [300/600], Loss: 1.369151 
#> Epoch [4/5], Step [400/600], Loss: 1.286297 
#> Epoch [4/5], Step [500/600], Loss: 1.342916 
#> Epoch [4/5], Step [600/600], Loss: 1.290759 
#> Epoch [5/5], Step [100/600], Loss: 1.198424 
#> Epoch [5/5], Step [200/600], Loss: 1.205739 
#> Epoch [5/5], Step [300/600], Loss: 1.192953 
#> Epoch [5/5], Step [400/600], Loss: 1.120249 
#> Epoch [5/5], Step [500/600], Loss: 1.196876 
#> Epoch [5/5], Step [600/600], Loss: 1.148688 
#> Epoch [6/5], Step [100/600], Loss: 1.056876 
#> Epoch [6/5], Step [200/600], Loss: 1.077124 
#> Epoch [6/5], Step [300/600], Loss: 1.069987 
#> Epoch [6/5], Step [400/600], Loss: 1.003438 
#> Epoch [6/5], Step [500/600], Loss: 1.092360 
#> Epoch [6/5], Step [600/600], Loss: 1.045942
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
#> Accuracy of the model on the 10000 test images: 83.090000 %
```

## Save the model


```r
# Save the model checkpoint
torch$save(model$state_dict(), 'model.ckpt')
```
