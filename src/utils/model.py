"""https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_simple.py"""
import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics


class Objective:
    def __init__(self, train_x, valid_x, train_y, valid_y):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y

    def __call__(self, trial):
        dtrain = lgb.Dataset(self.train_x, label=self.train_y)
        params = {
            "objective": "Regression",
            "metric": "mse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        gbm = lgb.train(params, dtrain)
        preds_val = gbm.predict(self.valid_x)
        preds_train = gbm.predict(self.train_x)
        val_loss = sklearn.metrics.mean_squared_error(self.valid_y, preds_val)
        train_loss = sklearn.metrics.mean_squared_error(self.train_y, preds_train)
        return val_loss  # train_loss
