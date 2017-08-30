## Notes

**The notebooks assume we receive data in the form of a generator function that yields mini-batches of numpy arrays**

**The notebooks are not specifically written for speed, instead they aim to create an easy comparison between the frameworks. However, any suggestions on improving the training-time are welcome**

**Notebooks are run on [Microsoft Azure Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.linux-data-science-vm-ubuntu?tab=Overview), where frameworks have been updated to the latest version**

## Goal

Create a Rosetta Stone of deep-learning frameworks to allow data-scientists to easily leverage their expertise from one framework to another (by translating, rather than learning from scratch). Also, to make the models more transparent to comparisons in terms of training-time and default-options.

A lot of online tutorials use very-low level APIs, which are very verbose, and don't make much sense (given higher-level helpers being available) for most use-cases unless one plans to create new layers. Here we try to apply the highest-level API possible, conditional on being to override conflicting defaults, to allow an easier comparison between frameworks. It will demonstrated that the code structure becomes very similar once higher-level APIs are used and can be roughly represented as:

- Load data; x_train, x_test, y_train, y_test = cifar_for_library(channel_first=?, one_hot=?)
- Generate CNN symbol (usually no activation on final dense-layer)
- Specify loss (cross-entropy comes bundles with softmax), optimiser and initialise weights + sessions
- Train on mini-batches from train-set using custom iterator (common data-source for all frameworks)
- Predict on fresh mini-batches from test-set
- Evaluate accuracy

Since we are essentially comparing a series of deterministic mathematical operations (albeit with a random initialization), it does not make sense to me to compare the accuracy across frameworks and instead they are reported as **checks we want to match**, to make sure we are comparing the same model architecture. 

## Results

### VGG-style CNN on CIFAR-10

| DL Library                               | Test Accuracy (%) | Training Time (s) |
| ---------------------------------------- | ----------------- | ----------------- |
| [Tensorflow (1.2.1)](Tensorflow_CIFAR.ipynb) | 72                | 300               |
| [CNTK (2.1)](CNTK_CIFAR.ipynb)           | 77                | 168               |
| [MXNet (0.11.0)](MXNet_CIFAR.ipynb)      | 75                | 153               |
| [PyTorch (0.2.0_1)](PyTorch_CIFAR.ipynb) | 73                | 351               |
| [Chainer (2.0.2)](Chainer_CIFAR.ipynb)   | 78                | 256               |
| [Keras (2.0.6) (TF)](Keras_TF_CIFAR.ipynb) | 77                | 408               |
| [Keras (2.0.6) (CNTK)](Keras_CNTK_CIFAR.ipynb) | 76                | 588               |
| [Caffe2](Caffe2_CIFAR.ipynb)             | 75                | 312               |


