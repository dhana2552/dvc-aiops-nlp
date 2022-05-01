import logging
import joblib
import scipy.sparse as sp
import numpy as np

def save_matrix(df, matrix, out_path):
    id_matrix = sp.csr_matrix(df.id.astype(np.int64)).T
    label_matrix = sp.csr_matrix(df.label.astype(np.int64)).T
    
    result = sp.hstack([id_matrix, label_matrix, matrix], format="csr")
    
    joblib.dump(result, out_path)
    msg = f"Saving matrix to {out_path} of size: {result.shape} and dtype: {result.dtype}"
    logging.info(msg)