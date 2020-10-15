import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
import utils

files = utils.filenames()
clf = GaussianNB()

d4 = lambda floatt: "{:.4f}".format(floatt)

def naive_bayes(dataset):
    X_train, y_train = utils.load_data(dataset['train'])
    X_val, y_val = utils.load_data(dataset['val'])
    X_test, y_test = utils.load_data(dataset['test'])

    model = clf.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    
    print()
    print(f'train acc {d4(train_acc)}')
    print(f'val acc   {d4(val_acc)}')
    print(f'test acc  {d4(test_acc)}')
    print()

    conf_matrix, label_matrix, macrof1, weightf1 = utils.compute_metrics(model, X_test, y_test)

    print(f'The confusion matrix\n{conf_matrix}\n')
    
    metric_str = ''
    for i, letter in enumerate(dataset['letters']):
        metric_str += f"{letter}: precision={d4(label_matrix[i]['precision'])} recall={d4(label_matrix[i]['recall'])} f1={d4(label_matrix[i]['f1'])}\n"

    print(f'{metric_str}\n')

    print(f'macro f1 = {d4(macrof1)}')
    print(f'weighted f1 = {d4(weightf1)}')


naive_bayes(files['greek'])
