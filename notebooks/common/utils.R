# Create an array of fake data to run inference on
give_fake_data <- function(batches){
  set.seed(0)
  dta <- array(runif(batches*224*224*3), dim = c(batches, 224, 224, 3))
  dta_swapped <- aperm(dta, c(1, 4, 3, 2))
  return(list(dta, dta_swapped))
}

# Get GPU name
get_gpu_name <- function(){
    tryCatch(
        {
            out_list <- system("nvidia-smi --query-gpu=gpu_name --format=csv", intern = TRUE)
            out_list <- out_list[out_list != "name"]
            return(out_list)
        },
        error = function(e)
        {
            print(e)
        }
        )
}

# Get CUDA version
get_cuda_version <- function(){
    tryCatch(
        {
            out <- system("cat /usr/local/cuda/version.txt", intern = TRUE)
            return(out)
        },
        error = function(e)
        {
            print(e)
        }
        )
}

# Get CuDNN version
get_cudnn_version <- function(){
    tryCatch(
        {
            out <- system("cat /usr/include/cudnn.h | grep CUDNN_MAJOR", intern = TRUE)[1]
            indx <- regexpr("(\\d+)", out)
            major <- regmatches(out, indx)
            
            out <- system("cat /usr/include/cudnn.h | grep CUDNN_MINOR", intern = TRUE)[1]
            indx <- regexpr("(\\d+)", out)
            minor <- regmatches(out, indx)
            
            out <- system("cat /usr/include/cudnn.h | grep CUDNN_PATCHLEVEL", intern = TRUE)[1]
            indx <- regexpr("(\\d+)", out)
            patch <- regmatches(out, indx)
            
            version <- paste(major, minor, patch, sep = ".")
            return(paste0("CuDNN Version ", version))
        },
        error = function(e)
        {
            print(e)
        }
        )
}

# Function to download the cifar data, if not already downloaded
maybe_download_cifar <- function(src = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz'){
  
  tryCatch(
    {
      data <- suppressWarnings(process_cifar_mat())
      return(data)
    },
    error = function(e)
    {
      print(paste0('Data does not exist. Downloading ', src))
      download.file(src, destfile="tmp.tar.gz")
      print('Extracting files ...')
      untar("tmp.tar.gz")
      file.remove('tmp.tar.gz')
      return(process_cifar_mat())
    }
  )
}

# A function to process CIFAR10 dataset
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

# A function to load CIFAR10 dataset
cifar_for_library <- function(channel_first = TRUE, one_hot = FALSE) {
  
  require(reticulate)
  
  cifar <- maybe_download_cifar()
  
  x_train <- cifar$x_train
  y_train <- cifar$y_train
  x_test <- cifar$x_test
  y_test <- cifar$y_test
  
  # Channels first or last
  if (channel_first){
    x_train <- array_reshape(x_train, c(50000, 3, 32, 32))
    x_test <- array_reshape(x_test, c(10000, 3, 32, 32))
  } else {
    x_train <- array_reshape(x_train, c(50000, 32, 32, 3))
    x_test <- array_reshape(x_test, c(10000, 32, 32, 3))
  }
  
  # One-hot encoding
  if(one_hot){
    Y = data.frame(label = factor(y_train))
    y_train = with(Y, model.matrix(~label+0))
    Y = data.frame(label = factor(y_test))
    y_test = with(Y, model.matrix(~label+0))
  }
  
  list(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
  
}

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


