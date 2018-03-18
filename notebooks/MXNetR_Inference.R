library(mxnet)

# Import util functions
source("./common/utils.R")

# Import hyper-parameters
params <- load_params("inference")

print(paste0("OS: ", Sys.info()["sysname"]))
print(R.version$version.string)
print(paste0("MXNet: ", packageVersion("mxnet")))

# Create batches of fake data
fake_data <- give_fake_data(params$BATCH_SIZE * params$BATCHES_GPU, col_major = TRUE)

cat('x_train shape:', dim(fake_data), '\n')

full_model <- load_resnet50()

all_layers <- full_model$symbol$get.internals()
tail(all_layers$outputs, 10)

fe_sym <- all_layers[[match("flatten0_output", all_layers$outputs)]] # https://github.com/apache/incubator-mxnet/issues/2535

model <- list(symbol = fe_sym,
               arg.params = full_model$arg.params,
               aux.params = full_model$aux.params)

class(model) <- "MXFeedForwardModel"

cold_start <- predict(
  model = model,
  X = fake_data,
  ctx = mx.gpu(0),
  array.batch.size = params$BATCH_SIZE,
  array.layout = "colmajor",
  allow.extra.params = TRUE
)

t <- system.time(
    features <- predict(
      model = model,
      X = fake_data,
      ctx = mx.gpu(0),
      array.batch.size = params$BATCH_SIZE,
      array.layout = "colmajor",
      allow.extra.params = TRUE
      )
  )

paste("Wall time:", t[3])

paste("Images per second", params$BATCH_SIZE * params$BATCHES_GPU / t[3])
