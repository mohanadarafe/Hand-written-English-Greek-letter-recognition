import preprocess
import numpy as np
import sklearn

english_training_filename = './data/Assig1-Dataset/train_1.csv'
english_test_filename = './data/Assig1-Dataset/test_with_label_1.csv'
english_validate_filename = './data/Assig1-Dataset/val_1.csv'

greek_training_filename = '/data/Assig1-Dataset/train_2.csv'

clf = sklearn.naive_bayes.GaussianNB()
X_train, Y_train = preprocess.convertFileToTrainingSet(english_training_filename)

X_val, Y_val = preprocess.convertFileToTrainingSet(english_training_filename)

X_test, Y_test = preprocess.convertFileToTrainingSet(english_test_filename)

model = clf.fit(X_train,Y_train)

train_acc  = model.score(X_train, Y_train)
val_acc = model.score(X_val, Y_val)
test_acc= model.score(X_test,Y_test)

print(f'train accuracy is {train_acc}')
print(f'validation accuracy is {val_acc}')
print(f'test accuracy is {test_acc} ')
























    #                         Y_test          test_prediction
    # true prositive  TP        1                   1
    # true negative   TN        
    # false positive  FP
    # false negative  FN


