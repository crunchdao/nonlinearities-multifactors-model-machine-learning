# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).


import pdb

import numpy as np
import optuna
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from utils.model import Objective


def main():

    data, target = make_regression(
        n_samples=1000, n_features=50
    )  # X: (1000, 50), y: (1000,)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    study = optuna.create_study(direction="minimize")
    study.optimize(Objective(train_x, valid_x, train_y, valid_y), n_trials=100)
    # Print results
    print(f"Best MSE: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # best_params = study.best_params
    # best_params['objective'] = 'regression'
    # best_model = lgb.train(best_params, lgb.Dataset(train_x, label=train_y))


if __name__ == "__main__":
    main()
