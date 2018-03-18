# Use reticulate to convert imdb npz file to Rds.
# Run npz2r.pynb first to download/process imdb data

#devtools::install_github("rstudio/reticulate")

library(reticulate)

## find python binary (run from within python session):
# import sys
# sys.executable

use_python('/home/anta/anaconda3/bin/python')

source_python("npz2r.py")
x_train <- get_data("x_train")
y_train <- get_data("y_train")
x_test <- get_data("x_test")
y_test <- get_data("y_test")

y_train <- as.integer(y_train)
y_test <- as.integer(y_test)

imdb <- list(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

saveRDS(imdb, file='imdb.Rds')
