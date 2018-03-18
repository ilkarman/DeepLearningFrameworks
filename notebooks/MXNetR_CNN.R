library(mxnet)

# Import util functions
source("./common/utils.R")

# Import hyper-parameters
params <- load_params("cnn")

print(paste0("OS: ", Sys.info()["sysname"]))
print(R.version$version.string)
print(paste0("MXnet: ", packageVersion("mxnet")))

create_symbol <- function(){
  
  data <- mx.symbol.Variable('data')
  # size = [(old-size - kernel + 2*padding)/stride]+1
  # if kernel = 3, pad with 1 either side
  conv1 <- mx.symbol.Convolution(data=data, num_filter=50, pad=c(1,1), kernel=c(3,3))
  relu1 <- mx.symbol.Activation(data=conv1, act_type="relu")
  conv2 <- mx.symbol.Convolution(data=relu1, num_filter=50, pad=c(1,1), kernel=c(3,3))
  pool1 <- mx.symbol.Pooling(data=conv2, pool_type="max", kernel=c(2,2), stride=c(2,2))
  relu2 <- mx.symbol.Activation(data=pool1, act_type="relu")
  drop1 <- mx.symbol.Dropout(data=relu2, p=0.25)
  
  conv3 <- mx.symbol.Convolution(data=drop1, num_filter=100, pad=c(1,1), kernel=c(3,3))
  relu3 <- mx.symbol.Activation(data=conv3, act_type="relu")
  conv4 <- mx.symbol.Convolution(data=relu3, num_filter=100, pad=c(1,1), kernel=c(3,3))
  pool2 <- mx.symbol.Pooling(data=conv4, pool_type="max", kernel=c(2,2), stride=c(2,2))
  relu4 <- mx.symbol.Activation(data=pool2, act_type="relu")
  drop2 <- mx.symbol.Dropout(data=relu4, p=0.25)
  
  flat1 <- mx.symbol.Flatten(data=drop2)
  fc1 <- mx.symbol.FullyConnected(data=flat1, num_hidden=512)
  relu7 <- mx.symbol.Activation(data=fc1, act_type="relu")
  drop4 <- mx.symbol.Dropout(data=relu7, p=0.5)
  fc2 <- mx.symbol.FullyConnected(data=drop4, num_hidden=params$N_CLASSES) 
  
  input_y <- mx.symbol.Variable('softmax_label')  
  mx.symbol.SoftmaxOutput(data=fc2, label=input_y, name="softmax")
  
}

cifar <- cifar_for_library()
x_train <- cifar$x_train
y_train <- cifar$y_train
x_test <- cifar$x_test
y_test <- cifar$y_test

rm(cifar)

cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')
cat('y_train shape:', length(y_train), '\n')
cat('y_test shape:', length(y_test), '\n')

plot_image(x_train[,,,1])

sym <- create_symbol()

if (params$GPU) {
  ctx = mx.gpu(0)
} else {
  ctx = mx.cpu()
}

train_iter <- mx.io.arrayiter(x_train, y_train, batch.size = params$BATCHSIZE, shuffle = TRUE)

start.time <- Sys.time()

model <- mx.model.FeedForward.create(
  symbol = sym,
  X = train_iter,
  ctx = ctx,
  num.round = params$EPOCHS,
  learning.rate = params$LR,
  momentum = params$MOMENTUM,
  eval.metric = mx.metric.accuracy,
  initializer = mx.init.Xavier(rnd_type = 'uniform'),
  epoch.end.callback = mx.callback.log.train.metric(100)
)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

start.time <- Sys.time()

y_guess <- predict(model, mx.io.arrayiter(x_test, y_test, batch.size = params$BATCHSIZE, shuffle = FALSE))
y_guess <- apply(y_guess, 2, function(x) which(x==max(x))-1)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

print(paste("Accuracy:", sum(y_guess == y_test)/length(y_guess)))
