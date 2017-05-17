from pickle import load
from os import listdir
from os.path import join, isfile

def read_CIFAR10(cifar_folder, nr_training, nr_test, nr_validation=0):
    filenames = [f for f in listdir(cifar_folder if isfile(f))
    data = dict
    for filename in filenames:
        print(filename)
        with open(join(cifar_folder, filename), 'rb') as fo:
            new_data = load(fo, encoding='bytes')
            print(new_data)
            data += load(fo, encoding='bytes')

    print(data)
    return data
