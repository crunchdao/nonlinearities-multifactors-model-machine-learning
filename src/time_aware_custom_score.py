# %%
import pandas as pd
import requests
import  csv
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn import model_selection, metrics 
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import csv
from scipy.stats import spearmanr 
import pdb


# %%
df = pd.read_parquet("../data/f_matrix.parquet")
target = pd.read_parquet("../data/target.parquet")
b_matrix = pd.read_parquet("../data/b_matrix.parquet")
target = target[['date', 'target_r']]
# Only way to get extra information in a custom scorer function in sklearn is to pass a pandas Serie with the needed data in index.
# https://stackoverflow.com/questions/67227646/can-i-get-extra-information-to-a-custom-scorer-function-in-sklearn
target = target.set_index('date')
target = target.squeeze()

# %%
class TimeSeriesSplitGroups(_BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)
    
    '''
    This function make sure that the split is not done arbitrarly in the middle of a cross-section.
    credit: https://forum.numer.ai/t/era-wise-time-series-cross-validation/791
    '''
    def split(self, X, y=None, groups=None):
        """
        Args:
            X: dataframe --> [50030 rows x 7 columns] -> no date column
            y: Pandas series --> date and target--> len 50030
            groups: Pandas series --> date --> len 50030
        
        """
        X, y, groups = indexable(X, y, groups) # moons
        n_samples = _num_samples(X) # 50030
        n_splits = self.n_splits # 5
        n_folds = n_splits + 1 # 6
        group_list = np.unique(groups) # all unique moons
        n_groups = len(group_list) # total unique moons --> 50

        if n_folds > n_groups: # trivial
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_groups))
        indices = np.arange(n_samples) # array([    0,     1,     2, ..., 50027, 50028, 50029])
        test_size = (n_groups // n_folds) # floor division of index --> 50/6 ~ 8
        test_starts = range(test_size + n_groups % n_folds, # range(10, 50, 8)
                            n_groups, test_size)
        test_starts = list(test_starts)[::-1] # [42, 34, 26, 18, 10]
        for test_start in test_starts: # [42, 34, 26, 18, 10] # 5 splits -->
            yield (indices[groups.isin(group_list[:test_start])],
                   indices[groups.isin(group_list[test_start:test_start + test_size])])

'''
Custom scorer function.
Cross-sectional Spearman's correlation Sharpe ratio.
'''
def spearman(y_true, y_pred):
    pdb.set_trace()
    data = pd.DataFrame(y_true)
    data['preds'] = y_pred
    data.reset_index(inplace=True)
    cor = data.groupby('date').corr(method="spearman").iloc[0::2,-1]
    
    return cor.mean()/cor.std()

error = []
times = []
def custom_loss(y_true, y_pred):
    pdb.set_trace()
    moon = str(y_true.index[0].date())
    b_moon = b_matrix[b_matrix.date == moon].drop(columns=["date"]).values
    y_hat = y_pred
    times.append(1)
    Omega = np.identity(len(y_hat)) - np.dot(b_moon,  np.linalg.pinv(b_moon))
    z_hat = np.dot(Omega, y_hat)
    #error.append(np.dot(z_hat - y_true, z_hat - y_true)/len(y_hat)) # inner_product/len --> mean squared error
    error.append(np.linalg.norm(z_hat - y_true))
    print(error[-1])
    return np.linalg.norm(y_hat - y_true)#0#error[-1]

features = [f for f in df.columns if f.startswith("feature")]
eras = df.date # all dates

# %%
cv_score = []
model = XGBRegressor(colsample_bytree=0.06, learning_rate=0.006, n_estimators=2000, max_depth=4, nthread=8, objective_fit= custom_loss)
del df['date']

# %%
# Timeserie nested k-fold
score = np.mean(model_selection.cross_val_score(
            model,
            df,
            target,
            cv=TimeSeriesSplitGroups(49), # increase granularity here
            n_jobs=1,
            verbose=1,
            groups=eras, # dates
            scoring=metrics.make_scorer(custom_loss, greater_is_better=False)))
print(score)
print(len(times))