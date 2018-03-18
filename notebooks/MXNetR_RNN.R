# https://jeremiedb.github.io/mxnet_R_bucketing/NLP_Classification_GPU.html
# http://dmlc.ml/rstats/2017/10/11/rnn-bucket-mxnet-R.html


library(mxnet)

# Import util functions
source("./common/utils.R")

# Import hyper-parameters
params <- load_params("lstm")

print(paste0("OS: ", Sys.info()["sysname"]))
print(R.version$version.string)
print(paste0("MXNet: ", packageVersion("mxnet")))

imdb <- imdb_for_library()
x_train <- imdb$x_train
y_train <- imdb$y_train
x_test <- imdb$x_test
y_test <- imdb$y_test
rm(imdb)

x_train <- aperm(x_train)
x_test <- aperm(x_test)

train_buckets <- list(`150`=list(data=x_train, label=y_train))
test_buckets <- list(`150`=list(data=x_test, label=y_test))

train_iter <- mx.io.bucket.iter(buckets = train_buckets, 
                                       batch.size = params$BATCHSIZE, 
                                       data.mask.element = 0, shuffle = TRUE)

test_iter <- mx.io.bucket.iter(buckets = test_buckets, 
                                batch.size = params$BATCHSIZE, 
                                data.mask.element = 0, shuffle = FALSE)


sym <- rnn.graph(
  num_rnn_layer = 1,
  input_size = params$MAXFEATURES, 
  num_embed = 2,# params$EMBEDSIZE, 
  num_hidden = 4,# params$NUMHIDDEN,
  num_decode = 2,
  dropout = 0,
  ignore_label = -1,
  bidirectional = FALSE, 
  loss_output = "softmax",
  config = "seq-to-one",
  cell_type = "gru",
  masking = TRUE, 
  output_last_state = FALSE, 
  rnn.state = NULL, 
  rnn.state.cell = NULL
  )

# mx.opt.adam in docs but not in pacakge
optimizer <- mx.opt.create(name = "adam", learning.rate = params$LR, beta1 = params$BETA_1, beta2 = params$BETA_2, epsilon = params$EPS)

initializer <- mx.init.Xavier(rnd_type = "uniform")

logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(period = 1, logger = logger)

system.time(
  model <- mx.model.buckets(symbol = sym,
                            train.data = train_iter,
                            num.round = params$EPOCHS, 
                            ctx = mx.gpu(0),
                            verbose = TRUE,
                            metric = mx.metric.accuracy,
                            optimizer = optimizer,  
                            initializer = initializer,
                            epoch.end.callback = epoch.end.callback)
)

system.time(
  infer <- mx.infer.rnn(infer.data = test_iter, model = model, ctx = mx.gpu(0))
)

pred_raw <- t(as.array(infer))
y_guess <- max.col(pred_raw, tie = "first") - 1
print(paste("Accuracy:", sum(y_guess == y_test)/length(y_guess)))

