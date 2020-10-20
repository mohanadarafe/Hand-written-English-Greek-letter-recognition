# Hand-written English & Greek letter recognition
https://github.com/mohanadarafe/comp472-a1

The following project is written in partial fulfilment of COMP472 : Artificial Intelligence.

## Project
The aim of the project is to take a set of 32x32 image hand-written letters in Greek & English and pass them to various machine learning models. We analyze the metrics & overall performance of each model.

## Requirements
### Conda
If you use Conda, you can simply run the environment setup for you. More instructions will be elaborated on the Setup.

### Not using Conda
If you do not have Conda, make sure you install the following dependencies:
- python 3.7
- scikit-learn
- matplotlib
- pandas
- numpy

## Setup
Make sure you have Conda installed on your machine
### Using Conda
```
conda env create --name project1 --file=env.yml
conda activate project1
python models/execute.py
```

### Other
```
pip3 install <dependency> # for all dependencies
python models/execute.py
```
