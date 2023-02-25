import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_kernels
from tqdm import tqdm

class kernel_ridge_regressor:

    def __init__(self, metric):
        self.metric = metric
        

    

    def calculate_coeff(self, data, C = 3, D = 0):
        """Calculates the kernel"""
        X_train, B_train, y_train = data
        K = pairwise_kernels(X_train, metric = self.metric)
        BBK = B_train.T @ B_train @ K
        M = K + C * np.identity(K.shape[0]) + D * BBK
        dual_coef = np.linalg.solve(M, y_train)

        return dual_coef


    def train(self, data, batch_size = 10, future_moons = 5, C = 3, D = 0, exposure = False):
        X_train, B_train, y_train, X_df, B_df, y_df, all_epochs = data
        dual_coef = self.calculate_coeff((X_train, B_train, y_train), C = C, D = D)
        
        B = B_df
        i = B.shape[1] - 1

        y_hat_df = pd.DataFrame()
        y_hat_df["date"] = X_df[X_df["date"].isin(all_epochs[batch_size : batch_size + future_moons])].date
        y_hat_df["y_hat"] = np.nan
        y_hat_df["y_test"] = np.nan

        

        for moon in tqdm(range(future_moons)):
            epochs = all_epochs[0 : batch_size + moon + 1] # train data


            i = B.shape[1] - 1
            j = 0
            for epoch in epochs: # to initialize Batch_b need i and j from this loop
                B_np = B[B["date"] == epoch].drop("date", axis=1).to_numpy().T
                j += B_np.shape[1]

            Batch_B = np.zeros((i, j)) 

            j = 0
            for epoch in epochs:
                B_np = B[B["date"] == epoch].drop("date", axis=1).to_numpy().T
                dj = B_np.shape[1]
                Batch_B[:, j : j + dj] = B_np

                j += dj
            

            B_test = Batch_B
            X_test = X_df[X_df["date"].isin(epochs)].drop("date", axis=1).to_numpy()
            y_test = y_df.iloc[: len(X_test)].to_numpy().ravel()
            K_hat = pairwise_kernels(X_test, X_train, metric = self.metric)
            y_hat = K_hat @ dual_coef
            exp_vec = B_test @ y_hat
            exp = np.dot(exp_vec, exp_vec)

            if exposure:
                print("factor_exposure",exp)

            y_hat_df.loc[y_hat_df["date"] == epochs[-1], "y_hat"] = y_hat[-dj:]
            y_hat_df.loc[y_hat_df["date"] == epochs[-1], "y_test"] = y_test[-dj:]
        

        return y_hat_df