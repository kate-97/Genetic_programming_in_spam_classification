import pandas as pd
from decision_tree import *
from genetic_programming import *
from run_algorithm import execute_genetic_algorithm, execute_and_evaluate_the_best
import numpy as np

SAME_SEED = 10


def read_columns(path):
    file = open(path, "r")

    lines = file.readlines()
    file.close()

    words = [line[:line.index(':')][10:] for line in lines[33:81]]
    chars = [line[:line.index(':')][10] for line in lines[81:87]]
    features = words + chars

    features += ['capital_average', 'capital_longest', 'capital_total', 'class']
    return features


if __name__ == '__main__':
    features = read_columns("data/spambase.names")

    df = pd.read_csv('data/spambase.data', names=features)
    np.random.seed(SAME_SEED)
    try:
        with open("logs/classification_data_41.txt", "a") as f:
            print("Started GP:")
            # execute_and_evaluate_the_best(df, 2000, 0.05, 400, 20, i=23, logf=f)
            # execute_and_evaluate_the_best(df, 5000, 0.05, 1000, 20, i=25,  logf=f)
            # execute_and_evaluate_the_best(df, 5000, 0.1, 1000, 20, i=27, logf=f)
            # execute_and_evaluate_the_best(df, 2000, 0.05, 400, 50, i=3, logf=f)
            # execute_and_evaluate_the_best(df, 2000, 0.1, 400, 50, i=4, logf=f)
            # execute_and_evaluate_the_best(df, 5000, 0.05, 1000, 20, i=39, logf=f)
            execute_and_evaluate_the_best(df, 7000, 0.05, 1500, 20, i=41, logf=f)

            # execute_genetic_algorithm(df, 2000, 0.05, 400, 20, f)
            # execute_genetic_algorithm(df, 5000,0.05,400,1, f)
            # execute_genetic_algorithm(df, 2000, 0.05, 400, 50, f)
            # execute_genetic_algorithm(df, 2000, 0.1, 400, 50, f)
            # TODO:execute_genetic_algorithm(df, 2500, 0.15, 500, 50, f)
    except IOError:
        print("Error while opening file")
