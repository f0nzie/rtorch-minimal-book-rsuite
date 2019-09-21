

# Working with data.table

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

## Load dataset


```r
## Dataset iteration batch settings
# folders where the images are located
train_data_path = '~/mnist_png_full/training/'
test_data_path  = '~/mnist_png_full/testing/'
```

## Read the datasets without normalization


```r
train_dataset = torchvision$datasets$ImageFolder(root = train_data_path, 
    transform = torchvision$transforms$ToTensor()
)

print(train_dataset)
#> Dataset ImageFolder
#>     Number of datapoints: 60000
#>     Root location: /home/msfz751/mnist_png_full/training/
```


## Using `data.table`


```r
library(data.table)
library(tictoc)
tic()

fun_list <- list(
    numel = c("numel"),
    sum   = c("sum",    "item"),
    mean  = c("mean",   "item"),
    std   = c("std",    "item"),
    med   = c("median", "item"),
    max   = c("max",    "item"),
    min   = c("min",    "item")
    )

idx <- seq(0L, 5999L)

fun_get_tensor <- function(x) py_get_item(train_dataset, x)[[1]]

stat_fun <- function(x, str_fun) {
  fun_var <- paste0("fun_get_tensor(x)", "$", str_fun, "()")
  sapply(idx, function(x) 
    ifelse(is.numeric(eval(parse(text = fun_var))),  # size return chracater
           eval(parse(text = fun_var)),              # all else are numeric
           as.character(eval(parse(text = fun_var)))))
}  

dt <- data.table(ridx = idx+1,
  do.call(data.table, 
          lapply(
            sapply(fun_list, function(x) paste(x, collapse = "()$")), 
            function(y) stat_fun(1, y)
          )
  )
)
```

Summary statistics:

```r
head(dt)
#>    ridx numel sum  mean   std med max min
#> 1:    1  2352 366 0.156 0.329   0   1   0
#> 2:    2  2352 284 0.121 0.297   0   1   0
#> 3:    3  2352 645 0.274 0.420   0   1   0
#> 4:    4  2352 410 0.174 0.355   0   1   0
#> 5:    5  2352 321 0.137 0.312   0   1   0
#> 6:    6  2352 654 0.278 0.421   0   1   0
```

Elapsed time per size of sample:

```r
toc()
#> 103.99 sec elapsed

#    60    1.266 sec elapsed
#   600   11.798 sec elapsed;
#  6000  119.256 sec elapsed;
# 60000 1117.619 sec elapsed
```

