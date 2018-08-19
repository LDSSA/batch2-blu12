import os
import numpy as np


def export_ratings(U, I, R):

    path = os.path.join('..', 'data', 'interim', 'ratings.csv')

    ratings = R.tocoo()

    uid = np.array([U[row] for row in ratings.row], dtype='O')
    iid = np.array([I[col] for col in ratings.col], dtype='O')

    data = ratings.data

    uid_ = uid.reshape(-1, 1)
    iid_ = iid.reshape(-1, 1)
    data_ = data.reshape(-1, 1)

    arr = np.hstack([uid_, iid_, data_])

    np.savetxt(path, arr, delimiter=',', header='uid,iid,rui', fmt="%s",
               comments='')
