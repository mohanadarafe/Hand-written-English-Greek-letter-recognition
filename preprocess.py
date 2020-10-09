# import numpy as np
import csv
import numpy as np
import matplotlib.pyplot as plt

# Converts training set to a list of tuples (image binary sequence, output)
def seperateTrainingSet(trainingSet):
    X_train = []
    Y_train = []
    for image in trainingSet:
        size = len(image)

        img = image[:size - 1]
        X_train.append(img)

        label = image[size - 1]
        Y_train.append(label)

    assert len(X_train) == len(Y_train), "X_train should be same length as Y_train"

    return (X_train, Y_train)


# generic use for all files
def convertFileToTrainingSet(fileName):
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
        print("File not found!")
    return training_set

if __name__ == "__main__":
    X_train, Y_train = convertFileToTrainingSet('./data/Assig1-Dataset/train_2.csv')
    img = np.array(X_train[15]).reshape(32,32)
    print(Y_train[15])
    plt.imshow(img)
    plt.show()
