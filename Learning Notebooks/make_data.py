import os
import numpy as np
from scipy.sparse import coo_matrix


def make_data():

    data_dir = os.path.join('..', 'data')

    users = make_users()
    items = make_items()

    clicks = make_ratings(path=os.path.join(data_dir, 'clicks.csv'))

    purchases = make_ratings(path=os.path.join(data_dir, 'purchases.csv'))

    return users, items, clicks, purchases


def make_ratings(path):

    users = make_users()
    items = make_items()

    users_ = read_array_from_csv(path, 'object', 0)
    items_ = read_array_from_csv(path, 'object', 1)

    rows = make_rows(users, users_)
    cols = make_cols(items, items_)

    nrows = users.shape[0]
    ncols = items.shape[0]

    shape = (nrows, ncols)

    data = np.ones(rows.size)

    return coo_matrix((data, (rows, cols)), shape=shape)


def make_users():
    path = os.path.join('..', 'data', 'users.csv')
    users = read_array_from_csv(path, 'object', 0)
    return users[users.argsort()]


def make_items():
    path = os.path.join('..', 'data', 'items.csv')
    items = read_array_from_csv(path, 'object', 0)
    return items[items.argsort()]


def read_array_from_csv(path, dtype, column):
    return np.genfromtxt(path, dtype=dtype, skip_header=True, usecols=[column],
                         delimiter=',')


def make_rows(users, users_):
    rows = [np.argwhere(users == u)[0, 0] for u in users_]
    return np.array(rows)


def make_cols(items, items_):
    cols = [np.argwhere(items == i)[0, 0] for i in items_]
    return np.array(cols)
