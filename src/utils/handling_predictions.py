import numpy as np
import pandas as pd


def spear(x):
    """
    Args:
        x: DataFrame
    """
    spear = x.corr(numeric_only=True, method="spearman")
    spear = spear.iloc[0, 1]
    return spear


def pear(x):
    """
    Args:
        x: DataFrame
    """
    pear = x.corr(numeric_only=True)
    pear = pear.iloc[0, 1]
    return pear


# def orthogonalize_predictions(B, y_hat_df, all_epochs, batch_size, future_moons):

#     y_hat_date = y_hat_df[["date", "y_hat"]]
#     b_matrix = B[B["date"].isin(all_epochs[batch_size : batch_size + future_moons])]
#     data = Data(f_matrix=y_hat_date, b_matrix=b_matrix)
#     data.orthogonalize()
#     y_hat_date = data.f_matrix
#     y_hat_df["y_hat"] = y_hat_date["y_hat"]
#     orth_spearman_history = y_hat_df.groupby("date").apply(lambda x: spear(x))
