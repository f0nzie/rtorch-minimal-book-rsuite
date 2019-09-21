
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

## Load dataset


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

### using `data.frame`


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
```

Summary statistics:

```r
head(df, 20)
#>    ridx                    size numel sum  mean   std med   max min
#> 1     1 torch.Size([3, 28, 28])  2352 366 0.156 0.329   0 1.000   0
#> 2     2 torch.Size([3, 28, 28])  2352 284 0.121 0.297   0 1.000   0
#> 3     3 torch.Size([3, 28, 28])  2352 645 0.274 0.420   0 1.000   0
#> 4     4 torch.Size([3, 28, 28])  2352 410 0.174 0.355   0 1.000   0
#> 5     5 torch.Size([3, 28, 28])  2352 321 0.137 0.312   0 1.000   0
#> 6     6 torch.Size([3, 28, 28])  2352 654 0.278 0.421   0 1.000   0
#> 7     7 torch.Size([3, 28, 28])  2352 496 0.211 0.374   0 1.000   0
#> 8     8 torch.Size([3, 28, 28])  2352 549 0.233 0.399   0 1.000   0
#> 9     9 torch.Size([3, 28, 28])  2352 449 0.191 0.365   0 1.000   0
#> 10   10 torch.Size([3, 28, 28])  2352 465 0.198 0.367   0 1.000   0
#> 11   11 torch.Size([3, 28, 28])  2352 383 0.163 0.338   0 1.000   0
#> 12   12 torch.Size([3, 28, 28])  2352 499 0.212 0.378   0 1.000   0
#> 13   13 torch.Size([3, 28, 28])  2352 313 0.133 0.309   0 0.996   0
#> 14   14 torch.Size([3, 28, 28])  2352 360 0.153 0.325   0 1.000   0
#> 15   15 torch.Size([3, 28, 28])  2352 435 0.185 0.358   0 0.996   0
#> 16   16 torch.Size([3, 28, 28])  2352 429 0.182 0.358   0 1.000   0
#> 17   17 torch.Size([3, 28, 28])  2352 596 0.254 0.408   0 1.000   0
#> 18   18 torch.Size([3, 28, 28])  2352 527 0.224 0.392   0 1.000   0
#> 19   19 torch.Size([3, 28, 28])  2352 303 0.129 0.301   0 1.000   0
#> 20   20 torch.Size([3, 28, 28])  2352 458 0.195 0.364   0 1.000   0
```

Elapsed time per size of sample:

```r
toc()
#> 11.608 sec elapsed
#    60   1.663s
#   600  13.5s
#  6000  54.321 sec;
# 60000 553.489 sec elapsed
```



