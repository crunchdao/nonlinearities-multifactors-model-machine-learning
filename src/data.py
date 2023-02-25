import numpy as np
import pandas as pd


class Data_nn3ml:
    def __init__(self, X, B, Y):
        self.X_df = X
        self.B_df = B
        self.y_df = Y
        self.all_epochs = B.date.unique()

    # def gaussianize_target

    def train_data(self, batch_size=10):
        epochs = self.all_epochs[:batch_size]
        X_train = (
            self.X_df[self.X_df["date"].isin(epochs)].drop("date", axis=1).to_numpy()
        )
        y_train = self.y_df.iloc[: len(X_train)].to_numpy().ravel()

        i = self.B_df.shape[1] - 1
        j = 0
        for epoch in epochs:  # to initialize Batch_b need i and j from this loop
            B_np = (
                self.B_df[self.B_df["date"] == epoch].drop("date", axis=1).to_numpy().T
            )
            j += B_np.shape[1]

        Batch_B = np.zeros((i, j))

        j = 0
        for epoch in epochs:
            B_np = (
                self.B_df[self.B_df["date"] == epoch].drop("date", axis=1).to_numpy().T
            )
            dj = B_np.shape[1]
            Batch_B[:, j : j + dj] = B_np

            j += dj
        B_train = Batch_B
        return X_train, B_train, y_train
