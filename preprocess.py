# import numpy as np
import csv
import numpy as np
import timeit

english_training_filename = './data/Assig1-Dataset/train_1.csv'
english_test_filename = './data/Assig1-Dataset/test_with_label_1.csv'
english_validate_filename = './data/Assig1-Dataset/val_1.csv'

greek_training_filename = './data/Assig1-Dataset/train_2.csv'
greek_test_filename = './data/Assig1-Dataset/test_with_label_2.csv'
greek_validate_filename = './data/Assig1-Dataset/val_2.csv'


def seperateTrainingSet(raw_data):
    '''
    Input: raw_data is a 2D list of ints. Each row is a list representing a line
        in the original csv file. 

    This func converts raw_data to the numpy matrix X and label vector y that is
    conventional in ML.
    '''

    X = []
    y = []

    for image in raw_data:

        img = image[:-1]
        X.append(img)

        label = image[-1]
        y.append(label)

    assert len(X) == len(y), "X should be same length as y"

    return (np.array(X), np.array(y))


# generic use for all files
def convertFileToTrainingSet(fileName):
    '''For a given filename, generate the X 2D-array and y label vector'''
    x = []
    training_set = []
    try:
        with open(fileName, 'r') as excelFile:
            lines = csv.reader(excelFile, delimiter=',')
            for line in lines:
                x = [int(el) for el in line]
                training_set.append(x)
            training_set = seperateTrainingSet(training_set)
    except FileNotFoundError:
        print(f"File not found: {fileName}")

    return training_set


def load_data(filename):
    X = np.loadtxt(filename, dtype=np.float64, delimiter=',', skiprows=1)
    y = X[:, -1]  # the last column contains the labels
    X = X[:, :-1]

    return X, y


if __name__ == '__main__':
    print(timeit.timeit(
        f'convertFileToTrainingSet(english_training_filename)',
        setup='from __main__ import convertFileToTrainingSet, english_training_filename',
        number=1))
    print(timeit.timeit(
        f'load_data(english_training_filename)',
        setup='from __main__ import load_data, english_training_filename',
        number=1))
