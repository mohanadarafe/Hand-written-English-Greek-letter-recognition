import numpy as np
import utils
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

files = utils.filenames()

def best_mlp(dataset):
    outputdir = f'{utils.get_project_root_dir()}/reports/'
    outputdir = 'Best-MLP-DS1' if len(dataset['letters']) == 26 else 'Best-MLP-DS2'
    language = 'english' if len(dataset['letters']) == 26 else 'greek'
    print()
    print(f"-"*60)
    print(f"Running Best MLP on the {language} alphabet")
    X_train, y_train = utils.load_data(dataset['train'])
    X_val, y_val = utils.load_data(dataset['val'])
    X_test, y_test = utils.load_data(dataset['test'])

    X_grid = np.concatenate((X_train, X_val))
    y_grid = np.concatenate((y_train, y_val))
    separation_boundary = [-1 for _ in y_train] + [0 for _ in y_val]
    ps = PredefinedSplit(separation_boundary)

    for train_index, val_index in ps.split():
        assert len(X_train) == len(train_index), f"length X_train and train_index should be equal"
        assert len(X_val) == len(val_index),     f"length X_val and val_index shoudl be equal"

    param_grid = {
        'activation': ['logistic', 'identity', 'tanh', 'relu'],
        'hidden_layer_sizes': [(700), (50, 100, 50, 100, 50, 100, 50, 100, 50, 50)],  # we decide this value
        'solver': ['adam', 'sgd']
    }
    clf = GridSearchCV(MLPClassifier(), param_grid=param_grid, cv=ps)

    model = clf.fit(X_grid, y_grid)

    print(f'Best params are {model.best_params_}')

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)

    print(f'train acc {utils.d4(train_acc)}')
    print(f'val acc   {utils.d4(val_acc)}')
    print(f'test acc  {utils.d4(test_acc)}')
    print()

    conf_matrix, label_matrix, macrof1, weightf1 = utils.compute_metrics(model, X_test, y_test)

    print(f'The confusion matrix\n{conf_matrix}\n')

    metric_str = ''
    for i, letter in enumerate(dataset['letters']):
        metric_str += f"{letter}: precision={utils.d4(label_matrix[i]['precision'])} \
            recall={utils.d4(label_matrix[i]['recall'])} \
            f1={utils.d4(label_matrix[i]['f1'])}\n"

    print(f'{metric_str}\n')

    print(f'macro f1 = {utils.d4(macrof1)}')
    print(f'weighted f1 = {utils.d4(weightf1)}\n')
    print(f'Writing these results in {outputdir}')
    utils.generate_report(dataset['nolabel'], outputdir, model, label_matrix, dataset, (train_acc, val_acc, test_acc, macrof1, weightf1), "Best MLP")

    print(f"-"*60)


best_mlp(files['english'])
best_mlp(files['greek'])
