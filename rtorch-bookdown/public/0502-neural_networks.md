
# A very simple neural network

## Introduction
Source: https://github.com/jcjohnson/pytorch-examples#pytorch-nn

In this example we use the torch `nn` package to implement our two-layer network:

## Select device


```r
library(rTorch)

device = torch$device('cpu')

# device = torch.device('cuda') # Uncomment this to run on GPU
```

* `N` is batch size; 
* `D_in` is input dimension;
* `H` is hidden dimension; 
* `D_out` is output dimension.

## Create the dataset


```r
torch$manual_seed(0)
#> <torch._C.Generator>

N <- 64L; D_in <- 1000L; H <- 100L; D_out <- 10L

# Create random Tensors to hold inputs and outputs
x = torch$randn(N, D_in, device=device)
y = torch$randn(N, D_out, device=device)
```


## Define the model
Use the `nn` package to define our model as a sequence of layers. `nn.Sequential` is a Module which contains other Modules, and applies them in sequence to produce its output. Each Linear Module computes output from input using a linear function, and holds internal Tensors for its weight and bias.
After constructing the model we use the `.to()` method to move it to the
desired device.


```r
model <- torch$nn$Sequential(
  torch$nn$Linear(D_in, H),              # first layer
  torch$nn$ReLU(),
  torch$nn$Linear(H, D_out))$to(device)  # output layer

print(model)
#> Sequential(
#>   (0): Linear(in_features=1000, out_features=100, bias=True)
#>   (1): ReLU()
#>   (2): Linear(in_features=100, out_features=10, bias=True)
#> )
```


## Loss function
The `nn` package also contains definitions of popular loss functions; in this case we will use Mean Squared Error (__MSE__) as our loss function. Setting `reduction='sum'` means that we are computing the *sum* of squared errors rather than the mean; this is for consistency with the examples above where we manually compute the loss, but in practice it is more common to use mean squared error as a loss by setting `reduction='elementwise_mean'`.


```r
loss_fn = torch$nn$MSELoss(reduction = 'sum')
```


## Iterate through batches


```r
learning_rate = 1e-4

for (t in 1:500) {
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Tensor of input data to the Module and it produces
  # a Tensor of output data.
  y_pred = model(x)

  # Compute and print loss. We pass Tensors containing the predicted and true
  # values of y, and the loss function returns a Tensor containing the loss.
  loss = loss_fn(y_pred, y)
  
  cat(t, "\t")
  cat(loss$item(), "\n")
  
  # Zero the gradients before running the backward pass.
  model$zero_grad()

  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Tensors with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss$backward()

  # Update the weights using gradient descent. Each parameter is a Tensor, so
  # we can access its data and gradients like we did before.
  with(torch$no_grad(), {
      for (param in iterate(model$parameters())) {
        # in Python this code is much simpler. In R we have to do some conversions
        
        # param$data <- torch$sub(param$data,
        #                         torch$mul(param$grad$float(),
        #                           torch$scalar_tensor(learning_rate)))
        
        param$data <- param$data - param$grad * learning_rate
      }
   })
}  
#> 1 	628 
#> 2 	585 
#> 3 	547 
#> 4 	513 
#> 5 	482 
#> 6 	455 
#> 7 	430 
#> 8 	406 
#> 9 	385 
#> 10 	364 
#> 11 	345 
#> 12 	328 
#> 13 	311 
#> 14 	295 
#> 15 	280 
#> 16 	265 
#> 17 	252 
#> 18 	239 
#> 19 	226 
#> 20 	214 
#> 21 	203 
#> 22 	192 
#> 23 	181 
#> 24 	172 
#> 25 	162 
#> 26 	153 
#> 27 	145 
#> 28 	137 
#> 29 	129 
#> 30 	122 
#> 31 	115 
#> 32 	109 
#> 33 	103 
#> 34 	96.9 
#> 35 	91.5 
#> 36 	86.3 
#> 37 	81.5 
#> 38 	76.9 
#> 39 	72.6 
#> 40 	68.5 
#> 41 	64.6 
#> 42 	61 
#> 43 	57.6 
#> 44 	54.3 
#> 45 	51.3 
#> 46 	48.5 
#> 47 	45.8 
#> 48 	43.2 
#> 49 	40.9 
#> 50 	38.6 
#> 51 	36.5 
#> 52 	34.5 
#> 53 	32.7 
#> 54 	30.9 
#> 55 	29.3 
#> 56 	27.8 
#> 57 	26.3 
#> 58 	24.9 
#> 59 	23.7 
#> 60 	22.4 
#> 61 	21.3 
#> 62 	20.2 
#> 63 	19.2 
#> 64 	18.2 
#> 65 	17.3 
#> 66 	16.5 
#> 67 	15.7 
#> 68 	14.9 
#> 69 	14.2 
#> 70 	13.5 
#> 71 	12.9 
#> 72 	12.3 
#> 73 	11.7 
#> 74 	11.1 
#> 75 	10.6 
#> 76 	10.1 
#> 77 	9.67 
#> 78 	9.24 
#> 79 	8.82 
#> 80 	8.42 
#> 81 	8.05 
#> 82 	7.69 
#> 83 	7.35 
#> 84 	7.03 
#> 85 	6.72 
#> 86 	6.43 
#> 87 	6.16 
#> 88 	5.9 
#> 89 	5.65 
#> 90 	5.41 
#> 91 	5.18 
#> 92 	4.97 
#> 93 	4.76 
#> 94 	4.57 
#> 95 	4.38 
#> 96 	4.2 
#> 97 	4.03 
#> 98 	3.87 
#> 99 	3.72 
#> 100 	3.57 
#> 101 	3.43 
#> 102 	3.29 
#> 103 	3.17 
#> 104 	3.04 
#> 105 	2.92 
#> 106 	2.81 
#> 107 	2.7 
#> 108 	2.6 
#> 109 	2.5 
#> 110 	2.41 
#> 111 	2.31 
#> 112 	2.23 
#> 113 	2.14 
#> 114 	2.06 
#> 115 	1.99 
#> 116 	1.91 
#> 117 	1.84 
#> 118 	1.77 
#> 119 	1.71 
#> 120 	1.65 
#> 121 	1.59 
#> 122 	1.53 
#> 123 	1.47 
#> 124 	1.42 
#> 125 	1.37 
#> 126 	1.32 
#> 127 	1.27 
#> 128 	1.23 
#> 129 	1.18 
#> 130 	1.14 
#> 131 	1.1 
#> 132 	1.06 
#> 133 	1.02 
#> 134 	0.989 
#> 135 	0.954 
#> 136 	0.921 
#> 137 	0.889 
#> 138 	0.858 
#> 139 	0.828 
#> 140 	0.799 
#> 141 	0.772 
#> 142 	0.745 
#> 143 	0.719 
#> 144 	0.695 
#> 145 	0.671 
#> 146 	0.648 
#> 147 	0.626 
#> 148 	0.605 
#> 149 	0.584 
#> 150 	0.564 
#> 151 	0.545 
#> 152 	0.527 
#> 153 	0.509 
#> 154 	0.492 
#> 155 	0.476 
#> 156 	0.46 
#> 157 	0.444 
#> 158 	0.43 
#> 159 	0.415 
#> 160 	0.402 
#> 161 	0.388 
#> 162 	0.375 
#> 163 	0.363 
#> 164 	0.351 
#> 165 	0.339 
#> 166 	0.328 
#> 167 	0.318 
#> 168 	0.307 
#> 169 	0.297 
#> 170 	0.287 
#> 171 	0.278 
#> 172 	0.269 
#> 173 	0.26 
#> 174 	0.252 
#> 175 	0.244 
#> 176 	0.236 
#> 177 	0.228 
#> 178 	0.221 
#> 179 	0.214 
#> 180 	0.207 
#> 181 	0.2 
#> 182 	0.194 
#> 183 	0.187 
#> 184 	0.181 
#> 185 	0.176 
#> 186 	0.17 
#> 187 	0.165 
#> 188 	0.159 
#> 189 	0.154 
#> 190 	0.149 
#> 191 	0.145 
#> 192 	0.14 
#> 193 	0.136 
#> 194 	0.131 
#> 195 	0.127 
#> 196 	0.123 
#> 197 	0.119 
#> 198 	0.115 
#> 199 	0.112 
#> 200 	0.108 
#> 201 	0.105 
#> 202 	0.102 
#> 203 	0.0983 
#> 204 	0.0952 
#> 205 	0.0923 
#> 206 	0.0894 
#> 207 	0.0866 
#> 208 	0.0838 
#> 209 	0.0812 
#> 210 	0.0787 
#> 211 	0.0762 
#> 212 	0.0739 
#> 213 	0.0716 
#> 214 	0.0693 
#> 215 	0.0672 
#> 216 	0.0651 
#> 217 	0.0631 
#> 218 	0.0611 
#> 219 	0.0592 
#> 220 	0.0574 
#> 221 	0.0556 
#> 222 	0.0539 
#> 223 	0.0522 
#> 224 	0.0506 
#> 225 	0.0491 
#> 226 	0.0476 
#> 227 	0.0461 
#> 228 	0.0447 
#> 229 	0.0433 
#> 230 	0.042 
#> 231 	0.0407 
#> 232 	0.0394 
#> 233 	0.0382 
#> 234 	0.0371 
#> 235 	0.0359 
#> 236 	0.0348 
#> 237 	0.0338 
#> 238 	0.0327 
#> 239 	0.0317 
#> 240 	0.0308 
#> 241 	0.0298 
#> 242 	0.0289 
#> 243 	0.028 
#> 244 	0.0272 
#> 245 	0.0263 
#> 246 	0.0255 
#> 247 	0.0248 
#> 248 	0.024 
#> 249 	0.0233 
#> 250 	0.0226 
#> 251 	0.0219 
#> 252 	0.0212 
#> 253 	0.0206 
#> 254 	0.02 
#> 255 	0.0194 
#> 256 	0.0188 
#> 257 	0.0182 
#> 258 	0.0177 
#> 259 	0.0171 
#> 260 	0.0166 
#> 261 	0.0161 
#> 262 	0.0156 
#> 263 	0.0151 
#> 264 	0.0147 
#> 265 	0.0142 
#> 266 	0.0138 
#> 267 	0.0134 
#> 268 	0.013 
#> 269 	0.0126 
#> 270 	0.0122 
#> 271 	0.0119 
#> 272 	0.0115 
#> 273 	0.0112 
#> 274 	0.0108 
#> 275 	0.0105 
#> 276 	0.0102 
#> 277 	0.00988 
#> 278 	0.00959 
#> 279 	0.0093 
#> 280 	0.00902 
#> 281 	0.00875 
#> 282 	0.00849 
#> 283 	0.00824 
#> 284 	0.00799 
#> 285 	0.00775 
#> 286 	0.00752 
#> 287 	0.0073 
#> 288 	0.00708 
#> 289 	0.00687 
#> 290 	0.00666 
#> 291 	0.00647 
#> 292 	0.00627 
#> 293 	0.00609 
#> 294 	0.00591 
#> 295 	0.00573 
#> 296 	0.00556 
#> 297 	0.0054 
#> 298 	0.00524 
#> 299 	0.00508 
#> 300 	0.00493 
#> 301 	0.00478 
#> 302 	0.00464 
#> 303 	0.0045 
#> 304 	0.00437 
#> 305 	0.00424 
#> 306 	0.00412 
#> 307 	0.00399 
#> 308 	0.00388 
#> 309 	0.00376 
#> 310 	0.00365 
#> 311 	0.00354 
#> 312 	0.00344 
#> 313 	0.00334 
#> 314 	0.00324 
#> 315 	0.00314 
#> 316 	0.00305 
#> 317 	0.00296 
#> 318 	0.00287 
#> 319 	0.00279 
#> 320 	0.00271 
#> 321 	0.00263 
#> 322 	0.00255 
#> 323 	0.00248 
#> 324 	0.0024 
#> 325 	0.00233 
#> 326 	0.00226 
#> 327 	0.0022 
#> 328 	0.00213 
#> 329 	0.00207 
#> 330 	0.00201 
#> 331 	0.00195 
#> 332 	0.00189 
#> 333 	0.00184 
#> 334 	0.00178 
#> 335 	0.00173 
#> 336 	0.00168 
#> 337 	0.00163 
#> 338 	0.00158 
#> 339 	0.00154 
#> 340 	0.00149 
#> 341 	0.00145 
#> 342 	0.00141 
#> 343 	0.00137 
#> 344 	0.00133 
#> 345 	0.00129 
#> 346 	0.00125 
#> 347 	0.00121 
#> 348 	0.00118 
#> 349 	0.00114 
#> 350 	0.00111 
#> 351 	0.00108 
#> 352 	0.00105 
#> 353 	0.00102 
#> 354 	0.000987 
#> 355 	0.000958 
#> 356 	0.000931 
#> 357 	0.000904 
#> 358 	0.000877 
#> 359 	0.000852 
#> 360 	0.000827 
#> 361 	0.000803 
#> 362 	0.00078 
#> 363 	0.000757 
#> 364 	0.000735 
#> 365 	0.000714 
#> 366 	0.000693 
#> 367 	0.000673 
#> 368 	0.000654 
#> 369 	0.000635 
#> 370 	0.000617 
#> 371 	0.000599 
#> 372 	0.000581 
#> 373 	0.000565 
#> 374 	0.000548 
#> 375 	0.000532 
#> 376 	0.000517 
#> 377 	0.000502 
#> 378 	0.000488 
#> 379 	0.000474 
#> 380 	0.00046 
#> 381 	0.000447 
#> 382 	0.000434 
#> 383 	0.000421 
#> 384 	0.000409 
#> 385 	0.000397 
#> 386 	0.000386 
#> 387 	0.000375 
#> 388 	0.000364 
#> 389 	0.000354 
#> 390 	0.000343 
#> 391 	0.000334 
#> 392 	0.000324 
#> 393 	0.000315 
#> 394 	0.000306 
#> 395 	0.000297 
#> 396 	0.000288 
#> 397 	0.00028 
#> 398 	0.000272 
#> 399 	0.000264 
#> 400 	0.000257 
#> 401 	0.000249 
#> 402 	0.000242 
#> 403 	0.000235 
#> 404 	0.000228 
#> 405 	0.000222 
#> 406 	0.000216 
#> 407 	0.000209 
#> 408 	0.000203 
#> 409 	0.000198 
#> 410 	0.000192 
#> 411 	0.000186 
#> 412 	0.000181 
#> 413 	0.000176 
#> 414 	0.000171 
#> 415 	0.000166 
#> 416 	0.000161 
#> 417 	0.000157 
#> 418 	0.000152 
#> 419 	0.000148 
#> 420 	0.000144 
#> 421 	0.00014 
#> 422 	0.000136 
#> 423 	0.000132 
#> 424 	0.000128 
#> 425 	0.000124 
#> 426 	0.000121 
#> 427 	0.000117 
#> 428 	0.000114 
#> 429 	0.000111 
#> 430 	0.000108 
#> 431 	0.000105 
#> 432 	0.000102 
#> 433 	9.87e-05 
#> 434 	9.59e-05 
#> 435 	9.32e-05 
#> 436 	9.06e-05 
#> 437 	8.8e-05 
#> 438 	8.55e-05 
#> 439 	8.31e-05 
#> 440 	8.07e-05 
#> 441 	7.84e-05 
#> 442 	7.62e-05 
#> 443 	7.41e-05 
#> 444 	7.2e-05 
#> 445 	6.99e-05 
#> 446 	6.79e-05 
#> 447 	6.6e-05 
#> 448 	6.41e-05 
#> 449 	6.23e-05 
#> 450 	6.06e-05 
#> 451 	5.89e-05 
#> 452 	5.72e-05 
#> 453 	5.56e-05 
#> 454 	5.4e-05 
#> 455 	5.25e-05 
#> 456 	5.1e-05 
#> 457 	4.96e-05 
#> 458 	4.82e-05 
#> 459 	4.68e-05 
#> 460 	4.55e-05 
#> 461 	4.42e-05 
#> 462 	4.3e-05 
#> 463 	4.18e-05 
#> 464 	4.06e-05 
#> 465 	3.94e-05 
#> 466 	3.83e-05 
#> 467 	3.72e-05 
#> 468 	3.62e-05 
#> 469 	3.52e-05 
#> 470 	3.42e-05 
#> 471 	3.32e-05 
#> 472 	3.23e-05 
#> 473 	3.14e-05 
#> 474 	3.05e-05 
#> 475 	2.96e-05 
#> 476 	2.88e-05 
#> 477 	2.8e-05 
#> 478 	2.72e-05 
#> 479 	2.65e-05 
#> 480 	2.57e-05 
#> 481 	2.5e-05 
#> 482 	2.43e-05 
#> 483 	2.36e-05 
#> 484 	2.29e-05 
#> 485 	2.23e-05 
#> 486 	2.17e-05 
#> 487 	2.11e-05 
#> 488 	2.05e-05 
#> 489 	1.99e-05 
#> 490 	1.94e-05 
#> 491 	1.88e-05 
#> 492 	1.83e-05 
#> 493 	1.78e-05 
#> 494 	1.73e-05 
#> 495 	1.68e-05 
#> 496 	1.63e-05 
#> 497 	1.59e-05 
#> 498 	1.54e-05 
#> 499 	1.5e-05 
#> 500 	1.46e-05
```

These two expression are equivalent, with the first being the long version natural way of doing it in __PyTorch__. The second is using the generics in R for subtraction, multiplication and scalar conversion.

```
param$data <- torch$sub(param$data,
                        torch$mul(param$grad$float(),
                          torch$scalar_tensor(learning_rate)))
}
```

```
param$data <- param$data - param$grad * learning_rate
```
