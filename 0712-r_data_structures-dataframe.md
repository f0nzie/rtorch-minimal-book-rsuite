
# Working with data.frame

## Load PyTorch libraries


```r
library(rTorch)

torch       <- import("torch")
torchvision <- import("torchvision")
nn          <- import("torch.nn")
transforms  <- import("torchvision.transforms")
dsets       <- import("torchvision.datasets")
builtins    <- import_builtins()
np          <- import("numpy")
```

## Dataset iteration batch settings


```r
# folders where the images are located
train_data_path = '~/mnist_png_full/training/'
test_data_path  = '~/mnist_png_full/testing/'
```



```r
# read the datasets without normalization
train_dataset = torchvision$datasets$ImageFolder(root = train_data_path, 
    transform = torchvision$transforms$ToTensor()
)

print(train_dataset)
#> Dataset ImageFolder
#>     Number of datapoints: 60000
#>     Root location: /home/msfz751/mnist_png_full/training/
```


## Summary statistics for tensors

## using `data.frame`


```r
library(tictoc)
tic()

fun_list <- list(
    size  = c("size"),
    numel = c("numel"),
    sum   = c("sum",    "item"),
    mean  = c("mean",   "item"),
    std   = c("std",    "item"),
    med   = c("median", "item"),
    max   = c("max",    "item"),
    min   = c("min",    "item")
    )

idx <- seq(0L, 599L)    # how many samples

fun_get_tensor <- function(x) py_get_item(train_dataset, x)[[1]]

stat_fun <- function(x, str_fun) {
  fun_var <- paste0("fun_get_tensor(x)", "$", str_fun, "()")
  sapply(idx, function(x) 
    ifelse(is.numeric(eval(parse(text = fun_var))),  # size return chracater
           eval(parse(text = fun_var)),              # all else are numeric
           as.character(eval(parse(text = fun_var)))))
}  

df <- data.frame(ridx = idx+1,      # index number for the sample
  do.call(data.frame, 
          lapply(
              sapply(fun_list, function(x) paste(x, collapse = "()$")), 
              function(y) stat_fun(1, y)
          )
  )
)
head(df)
#>   ridx                    size numel sum  mean   std med max min
#> 1    1 torch.Size([3, 28, 28])  2352 366 0.156 0.329   0   1   0
#> 2    2 torch.Size([3, 28, 28])  2352 284 0.121 0.297   0   1   0
#> 3    3 torch.Size([3, 28, 28])  2352 645 0.274 0.420   0   1   0
#> 4    4 torch.Size([3, 28, 28])  2352 410 0.174 0.355   0   1   0
#> 5    5 torch.Size([3, 28, 28])  2352 321 0.137 0.312   0   1   0
#> 6    6 torch.Size([3, 28, 28])  2352 654 0.278 0.421   0   1   0
toc()
#> 11.27 sec elapsed
# 59    1.663s
#   599  13.5s
#  5999  54.321 sec; 137.6s
# 59999 553.489 sec elapsed
```


