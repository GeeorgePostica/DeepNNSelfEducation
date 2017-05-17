from pickle import load
from os import listdir
from os.path import join, isfile, splitext

def read_CIFAR10(cifar_folder):
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

    return Xtr, Ytr, Xte, Xte

if __name__ == '__main__':
    read_CIFAR10('cifar-10-batches-py')
