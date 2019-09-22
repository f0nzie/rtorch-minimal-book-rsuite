
# Logistic Regression on MNIST digits. IDX images - Yunjey

> Restart RStudio before running

Source: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/logistic_regression/main.py


## Code in R
The code in R of this example can be found in Chapter \@ref(mnistdigits).


## Code in Python


```r
library(rTorch)
```





```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
```


```python
# Hyper-parameters 
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
```



```python
# MNIST dataset (images and labels)
# IDX format
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
```



```python
# Logistic regression model
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
```



```python
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28*28)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
#> Epoch [1/5], Step [100/600], Loss: 2.2214
#> Epoch [1/5], Step [200/600], Loss: 2.1097
#> Epoch [1/5], Step [300/600], Loss: 1.9834
#> Epoch [1/5], Step [400/600], Loss: 1.9353
#> Epoch [1/5], Step [500/600], Loss: 1.8226
#> Epoch [1/5], Step [600/600], Loss: 1.7723
#> Epoch [2/5], Step [100/600], Loss: 1.7138
#> Epoch [2/5], Step [200/600], Loss: 1.6539
#> Epoch [2/5], Step [300/600], Loss: 1.6344
#> Epoch [2/5], Step [400/600], Loss: 1.6265
#> Epoch [2/5], Step [500/600], Loss: 1.5405
#> Epoch [2/5], Step [600/600], Loss: 1.4648
#> Epoch [3/5], Step [100/600], Loss: 1.3308
#> Epoch [3/5], Step [200/600], Loss: 1.3706
#> Epoch [3/5], Step [300/600], Loss: 1.3376
#> Epoch [3/5], Step [400/600], Loss: 1.3485
#> Epoch [3/5], Step [500/600], Loss: 1.2913
#> Epoch [3/5], Step [600/600], Loss: 1.3771
#> Epoch [4/5], Step [100/600], Loss: 1.2499
#> Epoch [4/5], Step [200/600], Loss: 1.1708
#> Epoch [4/5], Step [300/600], Loss: 1.2234
#> Epoch [4/5], Step [400/600], Loss: 1.1204
#> Epoch [4/5], Step [500/600], Loss: 1.1811
#> Epoch [4/5], Step [600/600], Loss: 1.1054
#> Epoch [5/5], Step [100/600], Loss: 1.0072
#> Epoch [5/5], Step [200/600], Loss: 1.0944
#> Epoch [5/5], Step [300/600], Loss: 1.0523
#> Epoch [5/5], Step [400/600], Loss: 1.0789
#> Epoch [5/5], Step [500/600], Loss: 1.0850
#> Epoch [5/5], Step [600/600], Loss: 0.9936
```


```python
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
#> Accuracy of the model on the 10000 test images: 82 %
```



```python
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
```


