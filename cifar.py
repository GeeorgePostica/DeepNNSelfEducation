from pickle import load
import platform
import numpy as np
import os
# from os.path import join, isfile, splitext
# from os import listdir

# def read_CIFAR10(cifar_folder, train_ratio=0):
#     filenames = [f for f in listdir(cifar_folder) if (splitext(f)[1] != '.html' and splitext(f)[1] != '.meta')]
#     Xtr = Ytr = Xte = Yte = []
#     for filename in filenames:
#         with open(join(cifar_folder, filename), 'rb') as fo:
#             data = load(fo, encoding='bytes')
#             # print(filename)
#             # print(data)
#             batch_key = b'batch_label'
#             data_key = b'data'
#             labels_key = b'labels'
#             if str(data[batch_key]).startswith('training'):
#                 Xtr.append(data[data_key])
#                 Ytr.append(data[labels_key])
#             else:
#                 Xte.append(data[data_key])
#                 Yte.append(data[labels_key])
#     if 0 < train_ratio < 1:
#         X = Xtr + Xte
#         Y = Ytr + Yte
#         nr_train = int(len(X) * train_ratio)
#         # nr_test = len(X) - nr_train
#         Xtr = X[:nr_train]
#         Ytr = Y[:nr_train]
#         Xte = X[nr_train:]
#         Yte = Y[nr_train:]
#     return Xtr, Ytr, Xte, Yte


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return load(f)
    elif version[0] == '3':
        return load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }

# if __name__ == '__main__':
#     print(read_CIFAR10('cifar-10-batches-py'))
