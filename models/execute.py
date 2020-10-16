import naive_bayes
import base_decision_tree
import best_decision_tree
import perceptron
import base_mlp
import best_mlp
import utils

'''
The following executes all ML models to produce the outputs.
'''
files = utils.filenames()

for lang in ['english', 'greek']:
    naive_bayes.naive_bayes(files[lang])
    base_decision_tree.base_dt(files[lang])
    best_decision_tree.best_dt(files[lang])
    perceptron.perceptron(files[lang])
    base_mlp.base_mlp(files[lang])
    best_mlp.best_mlp(files[lang])
