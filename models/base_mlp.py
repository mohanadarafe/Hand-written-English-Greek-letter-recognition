import numpy as np
import utils
from sklearn.neural_network import MLPClassifier

files = utils.filenames()

def base_mlp(dataset):
    outputdir = f'{utils.get_project_root_dir()}/reports/'
    outputdir += 'Base-MLP-DS1' if len(dataset['letters']) == 26 else 'Base-MLP-DS2'
    language = 'english' if len(dataset['letters']) == 26 else 'greek'
    print()
    print(f"-"*60)
    print(f"Running base MLP on the {language} alphabet")
    X_train, y_train = utils.load_data(dataset['train'])
    X_val, y_val = utils.load_data(dataset['val'])
    X_test, y_test = utils.load_data(dataset['test'])

    clf = MLPClassifier(activation="logistic", solver="sgd")
    model = clf.fit(X_train, y_train)

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
        metric_str += f"{letter}: precision={utils.d4(label_matrix[i]['precision'])} recall={utils.d4(label_matrix[i]['recall'])} f1={utils.d4(label_matrix[i]['f1'])}\n"

    print(f'{metric_str}\n')

    print(f'macro f1 = {utils.d4(macrof1)}')
    print(f'weighted f1 = {utils.d4(weightf1)}\n')
    print(f'Writing these results in {outputdir}')
    utils.generate_report(dataset['nolabel'], outputdir, model, label_matrix, dataset, (train_acc ,val_acc, test_acc, macrof1, weightf1), "Base MLP")

    print(f"-"*60)


if __name__=='__main__':
    base_mlp(files['english'])
    base_mlp(files['greek'])