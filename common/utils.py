import numpy as np
import os
import tarfile
import pickle
import subprocess
import sys
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

if sys.version_info.major == 2:
    # Backward compatibility with python 2.
    from six.moves import urllib
    urlretrieve = urllib.request.urlretrieve
else:
    from urllib.request import urlretrieve


def get_gpu_name():
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


def read_batch(src):
    '''Unpack the pickle files
    '''
    with open(src, 'rb') as f:
        if sys.version_info.major == 2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
    return data


def shuffle_data(X, y):
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y


def yield_mb(X, y, batchsize=64, shuffle=False):
    assert len(X) == len(y)
    if shuffle:
        X, y = shuffle_data(X, y)
    # Only complete batches are submitted
    for i in range(len(X) // batchsize):
        yield X[i * batchsize:(i + 1) * batchsize], y[i * batchsize:(i + 1) * batchsize]


def process_cifar():
    '''Load data into RAM'''
    print('Preparing train set...')
    train_list = [read_batch('./cifar-10-batches-py/data_batch_{0}'.format(i + 1)) for i in range(5)]
    x_train = np.concatenate([t['data'] for t in train_list])
    y_train = np.concatenate([t['labels'] for t in train_list])
    print('Preparing test set...')
    tst = read_batch('./cifar-10-batches-py/test_batch')
    x_test = tst['data']
    y_test = np.asarray(tst['labels'])
    return x_train, x_test, y_train, y_test


def maybe_download_cifar(src="https://ikpublictutorial.blob.core.windows.net/deeplearningframeworks/cifar-10-python.tar.gz"):
    '''Load the training and testing data
    Mirror of: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    '''
    try:
        return process_cifar()
    except:
        # Catch the exception that file doesn't exist
        # Download
        print('Data does not exist. Downloading ' + src)
        fname, h = urlretrieve(src, './delete.me')
        print('Extracting files...')
        with tarfile.open(fname) as tar:
            tar.extractall()
        os.remove(fname)
        return process_cifar()


def process_imdb():
    '''Load data into RAM'''
    with np.load('imdb.npz') as f:
        print('Preparing train set...')
        x_train, y_train = f['x_train'], f['y_train']
        print('Preparing test set...')
        x_test, y_test = f['x_test'], f['y_test']
    return x_train, x_test, y_train, y_test


def maybe_download_imdb(src="https://ikpublictutorial.blob.core.windows.net/deeplearningframeworks/imdb.npz"):
    '''Load the training and testing data
    Mirror of: https://s3.amazonaws.com/text-datasets/imdb.npz'''
    try:
        return process_imdb()
    except:
        # Catch exception that file doesn't exist
        # Download
        print('Data does not exist. Downloading ' + src)
        fname, h = urlretrieve(src, './imdb.npz')
        # No need to extract
        x_train, x_test, y_train, y_test = process_imdb()
        return x_train, x_test, y_train, y_test


def cifar_for_library(channel_first=True, one_hot=False):
    # Raw data
    x_train, x_test, y_train, y_test = maybe_download_cifar()
    # Scale pixel intensity
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Reshape
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)
    # Channel last
    if not channel_first:
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
    # One-hot encode y
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test


def imdb_for_library(seq_len=100, max_features=20000, one_hot=False):
    ''' Replicates same pre-processing as:
    https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py

    I'm not sure if we want to load another version of IMDB that has got 
    words, but if it does have words we would still convert to index in this 
    backend script that is not meant for others to see ...    

    But I'm worried this obfuscates the data a bit?
    '''
    # 0 (padding), 1 (start), 2 (OOV)
    START_CHAR = 1
    OOV_CHAR = 2
    INDEX_FROM = 3
    # Raw data (has been encoded into words already)
    x_train, x_test, y_train, y_test = maybe_download_imdb()
    # Combine for processing
    idx = len(x_train)
    _xs = np.concatenate([x_train, x_test])
    # Words will start from INDEX_FROM (shift by 3)
    _xs = [[START_CHAR] + [w + INDEX_FROM for w in x] for x in _xs]
    # Max-features - replace words bigger than index with oov_char
    # E.g. if max_features = 5 then keep 0, 1, 2, 3, 4 i.e. words 3 and 4
    if max_features:
        print("Trimming to {} max-features".format(max_features))
        _xs = [[w if (w < max_features) else OOV_CHAR for w in x] for x in _xs]
        # Pad to same sequences
    print("Padding to length {}".format(seq_len))
    xs = np.zeros((len(_xs), seq_len), dtype=np.int)
    for o_idx, obs in enumerate(_xs):
        # Match keras pre-processing of taking last elements
        obs = obs[-seq_len:]
        for i_idx in range(len(obs)):
            if i_idx < seq_len:
                xs[o_idx][i_idx] = obs[i_idx]
    # One-hot
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = np.array(xs[:idx]).astype(np.int32)
    x_test = np.array(xs[idx:]).astype(np.int32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test