import pandas as pd
import pdb as pdb

X = pd.read_parquet("X_HKML.parquet") # [345147 rows x 161 columns]
B = pd.read_parquet("B_HKML.parquet") # [345147 rows x 86 columns]
XB = pd.read_parquet("XB_PCA_HKML.parquet") # [310704 rows x 178 columns]
Y = pd.read_parquet("Y_HKML.parquet") # [345147 rows x 1 columns]

pdb.set_trace()