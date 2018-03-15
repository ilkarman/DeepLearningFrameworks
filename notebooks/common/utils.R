# Create an array of fake data to run inference on
give_fake_data <- function(batches){
  set.seed(0)
  dta <- array(runif(batches*224*224*3), dim = c(batches, 224, 224, 3))
  dta_swapped <- aperm(dta, c(1, 4, 3, 2))
  return(list(dta, dta_swapped))
}


# Function to download the cifar data, if not already downloaded
maybe_download_cifar <- function(col_major = TRUE, src = 'https://ikpublictutorial.blob.core.windows.net/deeplearningframeworks/cifar-10-binary.tar.gz '){
  
  tryCatch(
    {
      data <- suppressWarnings(process_cifar_bin(col_major))
      return(data)
    },
    error = function(e)
    {
      print(paste0('Data does not exist. Downloading ', src))
      download.file(src, destfile="tmp.tar.gz")
      print('Extracting files ...')
      untar("tmp.tar.gz")
      file.remove('tmp.tar.gz')
      return(process_cifar_bin(col_major))
    }
  )
}

# A function to process CIFAR10 dataset in matlab format
process_cifar_mat <- function(){
  
  require(R.matlab)
  
  train_labels <- list()
  train_data <- list()
  
  print('Preparing train set ...')
  for (i in seq(5)) {
    train <- readMat(paste0('./cifar-10-batches-mat/data_batch_', i, '.mat'))
    train_data[[i]] <- train$data
    train_labels[[i]] <- train$labels
  }
  
  x_train <- do.call(rbind, train_data)
  x_train <- x_train / 255
  y_train <- do.call(rbind, train_labels)
  
  
  print('Preparing test set ...')
  test <- readMat('./cifar-10-batches-mat/test_batch.mat')
  
  x_test <- test$data
  x_test <- x_test / 255
  y_test <- test$labels
  
  list(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
  
}

read_image <- function(i, to_read) {
  label <- readBin(to_read, integer(), n = 1, size = 1)
  image <- as.integer(readBin(to_read, raw(), size = 1, n = 32*32*3))
  list(label = label, image = image)
}

read_file <- function(f) {
  to_read <- file(f, "rb")
  examples <- lapply(1:10000, read_image, to_read)
  close(to_read)
  examples
}

# A function to process CIFAR10 dataset in binary format
process_cifar_bin <- function(col_major) {
  
  data_dir <- "cifar-10-batches-bin"
  
  train <- lapply(file.path(data_dir, paste0("data_batch_", 1:5, ".bin")), read_file)
  train <- do.call(c, train)
  
  x_train <- unlist(lapply(train, function(x) x$image))
  if (col_major) {
    perm <- c(2, 1, 3, 4)
  } else {
    perm <- c(4, 3, 2, 1)
  }
  
  x_train <- aperm(array(x_train, c(32, 32, 3, 50000)), perm = perm)
  x_train <- x_train / 255
  y_train <- unlist(lapply(train, function(x) x$label))
  
  test <- read_file(file.path(data_dir, "test_batch.bin"))
  x_test <- unlist(lapply(test, function(x) x$image))
  x_test <- aperm(array(x_test, c(32, 32, 3, 10000)), perm = perm)
  x_test <- x_test / 255
  y_test <- unlist(lapply(test, function(x) x$label))
  
  list(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)
}


# A function to load CIFAR10 dataset
cifar_for_library <- function(one_hot = FALSE, col_major = TRUE) {
  
  cifar <- maybe_download_cifar(col_major)
  
  x_train <- cifar$x_train
  y_train <- cifar$y_train
  x_test <- cifar$x_test
  y_test <- cifar$y_test
  
  if(one_hot){
    Y = data.frame(label = factor(y_train))
    y_train = with(Y, model.matrix(~label+0))
    Y = data.frame(label = factor(y_test))
    y_test = with(Y, model.matrix(~label+0))
  }
  
  list(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
  
}

# A function to load CIFAR10 dataset
# cifar_for_library <- function(channel_first = TRUE, one_hot = FALSE) {
#   
#   require(reticulate)
#   
#   cifar <- maybe_download_cifar()
#   
#   x_train <- cifar$x_train
#   y_train <- cifar$y_train
#   x_test <- cifar$x_test
#   y_test <- cifar$y_test
#   
#   # Channels first or last
#   if (channel_first){
#     x_train <- array_reshape(x_train, c(50000, 3, 32, 32))
#     x_test <- array_reshape(x_test, c(10000, 3, 32, 32))
#   } else {
#     x_train <- array_reshape(x_train, c(50000, 32, 32, 3))
#     x_test <- array_reshape(x_test, c(10000, 32, 32, 3))
#   }
#   
#   # One-hot encoding
#   if(one_hot){
#     Y = data.frame(label = factor(y_train))
#     y_train = with(Y, model.matrix(~label+0))
#     Y = data.frame(label = factor(y_test))
#     y_test = with(Y, model.matrix(~label+0))
#   }
#   
#   list(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
#   
# }

# Load hyper-parameters for different scenarios:
# cnn, lstm, or inference
load_params <- function(params_for){
    
    require(rjson)
    params <- fromJSON(file = "./common/params.json")

    if (params_for == "cnn"){
        return(params$params_cnn)
    } else if (params_for == "lstm"){
        return(params$params_lstm)
    } else if (params_for == "inference"){
        return(params$params_inf)
    } else {
        stop("params_for should be set to one of the following: cnn, lstm or inference.")
    }
}

# Plot a CIFAR10 image
plot_image <- function(img) {
  library(grid)
  img_dim <- dim(img)
  if (img_dim[1] < img_dim[3]) {
    r <- img[1,,]
    g <- img[2,,]
    b <- img[3,,]
  } else {
    r <- img[,,1]
    g <- img[,,2]
    b <- img[,,3]
  }
  img.col.mat <- rgb(r, g, b, maxColorValue = 1)
  dim(img.col.mat) <- dim(r)
  grid.raster(img.col.mat, interpolate = FALSE)
  rm(img.col.mat)
}

