import pandas as pd
from decision_tree import *
from genetic_programming import *
from run_algorithm import execute_genetic_algorithm, execute_and_evaluate_the_best
import numpy as np


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

    try:
        with open("logs/classification_data_7.txt", "a") as f:
            print("Started GP:")
            # execute_and_evaluate_the_best(df, 2000, 0.05, 400, 20, i=1, logf=f)
            # execute_and_evaluate_the_best(df, 5000, 0.05, 400, 1, i=2,  logf=f)
            # execute_and_evaluate_the_best(df, 2000, 0.05, 400, 50, i=3, logf=f)
            # execute_and_evaluate_the_best(df, 2000, 0.1, 400, 50, i=4, logf=f)
            execute_and_evaluate_the_best(df, 1500, 0.05, 200, 2, i=7, logf=f)

            # execute_genetic_algorithm(df, 2000, 0.05, 400, 20, f)
            # execute_genetic_algorithm(df, 5000,0.05,400,1, f)
            # execute_genetic_algorithm(df, 2000, 0.05, 400, 50, f)
            # execute_genetic_algorithm(df, 2000, 0.1, 400, 50, f)
            # TODO:execute_genetic_algorithm(df, 2500, 0.15, 500, 50, f)
    except IOError:
        print("Error while opening file")
