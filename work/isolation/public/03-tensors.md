
# Tensors

We describe the most important PyTorch methods in this chapter.

## Arithmetic of tensors

## Boolean operations

## Slicing




## Example
The following example was converted from PyTorch to rTorch to show differences and similarities of both approaches. The original source can be found here:

Source: https://github.com/jcjohnson/pytorch-examples#pytorch-tensors

### Load the libraries


```r
library(rTorch)

device = torch$device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

torch$manual_seed(0)
#> <torch._C.Generator>
```


* `N` is batch size; 
* `D_in` is input dimension;
* `H` is hidden dimension; 
* `D_out` is output dimension.
 
### Datasets 
We will create a random dataset for a two layer neural network.


```r
N <- 64L; D_in <- 1000L; H <- 100L; D_out <- 10L

# Create random Tensors to hold inputs and outputs
x <- torch$randn(N, D_in, device=device)
y <- torch$randn(N, D_out, device=device)
```



```r
# Randomly initialize weights
w1 <- torch$randn(D_in, H, device=device)   # layer 1
w2 <- torch$randn(H, D_out, device=device)  # layer 2
```

### Run the model


```r
learning_rate = 1e-6

# loop
for (t in 1:50) {
  # Forward pass: compute predicted y
  h <- x$mm(w1)
  h_relu <- h$clamp(min=0)
  y_pred <- h_relu$mm(w2)

  # Compute and print loss; loss is a scalar, and is stored in a PyTorch Tensor
  # of shape (); we can get its value as a Python number with loss.item().
  loss <- (torch$sub(y_pred, y))$pow(2)$sum()
  cat(t, "\t")
  cat(loss$item(), "\n")

  # Backprop to compute gradients of w1 and w2 with respect to loss
  grad_y_pred <- torch$mul(torch$scalar_tensor(2.0), torch$sub(y_pred, y))
  grad_w2 <- h_relu$t()$mm(grad_y_pred)
  grad_h_relu <- grad_y_pred$mm(w2$t())
  grad_h <- grad_h_relu$clone()
  # grad_h[h < 0] = 0
  mask <- grad_h$lt(0)
  # print(mask)
  # negatives <- torch$masked_select(grad_h, mask)
  # print(negatives)
  # negatives <- 0.0
  
  torch$masked_select(grad_h, mask)$fill_(0.0)
  
  # print(grad_h)
  grad_w1 <- x$t()$mm(grad_h)
   
  # Update weights using gradient descent
  w1 <- torch$sub(w1, torch$mul(learning_rate, grad_w1))
  w2 <- torch$sub(w2, torch$mul(learning_rate, grad_w2))
}  
#> 1 	29428666 
#> 2 	22572578 
#> 3 	20474034 
#> 4 	19486618 
#> 5 	1.8e+07 
#> 6 	15345387 
#> 7 	1.2e+07 
#> 8 	8557820 
#> 9 	5777508 
#> 10 	3791835 
#> 11 	2494379 
#> 12 	1679618 
#> 13 	1176170 
#> 14 	858874 
#> 15 	654740 
#> 16 	517359 
#> 17 	421628 
#> 18 	351479 
#> 19 	298321 
#> 20 	256309 
#> 21 	222513 
#> 22 	194530 
#> 23 	171048 
#> 24 	151092 
#> 25 	134001 
#> 26 	119256 
#> 27 	106431 
#> 28 	95220 
#> 29 	85393 
#> 30 	76739 
#> 31 	69099 
#> 32 	62340 
#> 33 	56344 
#> 34 	51009 
#> 35 	46249 
#> 36 	41992 
#> 37 	38182 
#> 38 	34770 
#> 39 	31705 
#> 40 	28946 
#> 41 	26458 
#> 42 	24211 
#> 43 	22179 
#> 44 	20334 
#> 45 	18659 
#> 46 	17138 
#> 47 	15753 
#> 48 	14494 
#> 49 	13347 
#> 50 	12301
```
