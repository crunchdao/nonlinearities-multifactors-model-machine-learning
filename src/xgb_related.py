import numpy as np
import xgboost as xgb

#from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd
import pdb
from tqdm import tqdm


error = []
def custom_loss(y_true, data):

    print(" handling moon", moon)
    y_hat = data.get_label()
    Omega = np.identity(len(y_hat)) - np.dot(b_moon,  np.linalg.pinv(b_moon))
    z_hat = np.dot(Omega, y_hat)
    
    error.append(np.dot(z_hat - y_true, z_hat - y_true)/len(y_hat))
    print(error[-1])
    
    grad = 2*np.dot(Omega, np.dot(Omega, y_hat) - y_true)/len(y_hat)
    hess = np.ones_like(grad) # 2*np.dot(Omega, Omega)
    return grad, hess

f_matrix = pd.read_parquet("/Users/utkarshpratiush/Cr_D/HKML/HKML/data/f_matrix.parquet") # [50030 rows x 8 columns]
b_matrix = pd.read_parquet("/Users/utkarshpratiush/Cr_D/HKML/HKML/data/b_matrix.parquet") # [50030 rows x 85 columns]
targets = pd.read_parquet("/Users/utkarshpratiush/Cr_D/HKML/HKML/data/target.parquet") # [50030 rows x 5 columns]

moons = f_matrix.date.unique()

# # Create minibatches of data
# batch_size = 32
# num_batches = int(np.ceil(X.shape[0] / batch_size))
# batches = [(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
#            for i in range(num_batches)]

# Load xgboost model

params = {'objective': 'reg:linear', 'verbose': True}
#params = {'objective': custom_loss, 'verbose': True}
dmatrix_start = xgb.DMatrix(f_matrix.drop(columns=["date"]).values, label=targets.drop(columns=["date"]).values[:, -1])

model_1 = xgb.train(params,  dmatrix_start,  1, num_boost_round=1)
#pdb.set_trace()
model_1.save_model('model_1.model')
# Define custom loss as a scorer for cross-validation
custom_scorer = make_scorer(custom_loss, greater_is_better=False)

# Train for 5 epochs
num_epochs = 10
learning_rate = 0.1
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    for i,moon in tqdm(enumerate(moons)):
        X_moon = f_matrix[f_matrix.date == moon].drop(columns=["date"]).values
        y_moon = targets[targets.date == moon].drop(columns=["date"]).values[:, -1].ravel()

        b_moon = b_matrix[b_matrix.date == moon].drop(columns=["date"]).values
        dmatrix = xgb.DMatrix(X_moon, label=y_moon)
        model_2_v2 = xgb.train(params, dmatrix, xgb_model='model_1.model',num_boost_round=1, obj=custom_loss)
        model_2_v2.save_model('model_1.model')
        #if i % 10 == 0:
        #    print(f"Epoch {epoch}, batch {i}, loss: {model.eval(dmatrix)}")

pdb.set_trace()
# Evaluate model using cross-validation and custom loss scorer
#scores = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
#print(f"Custom loss score: {np.mean(scores)}")
