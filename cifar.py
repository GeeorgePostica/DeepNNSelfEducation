from pickle import load
from os import listdir
from os.path import join, isfile, splitext

def read_CIFAR10(cifar_folder, train_ratio=0):
    filenames = [f for f in listdir(cifar_folder) if (splitext(f)[1] != '.html' and splitext(f)[1] != '.meta')]
    Xtr = Ytr = Xte = Yte = []
    for filename in filenames:
        with open(join(cifar_folder, filename), 'rb') as fo:
            data = load(fo, encoding='bytes')
            print(filename)
            print(data)
            if data['batch_label'].startswith('training'):
                Xtr.append(data['data'])
                Ytr.append(data['labels'])
            else:
                Xte.append(data['data'])
                Yte.append(data['labels'])
    if 0 < train_ratio < 1:
        X = Xtr + Xte
        Y = Ytr + Yte
        nr_train = int(len(X) * train_ratio)
        # nr_test = len(X) - nr_train
        Xtr = X[:nr_train]
        Ytr = Y[:nr_train]
        Xte = X[nr_train:]
        Yte = Y[nr_train:]
    return Xtr, Ytr, Xte, Yte

if __name__ == '__main__':
    read_CIFAR10('cifar-10-batches-py')
