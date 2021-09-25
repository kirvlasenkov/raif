import pandas as pd
import numpy as np

def bin_column(column, bins):
    ans = np.array([len(bins)] * column.shape[0])
    k = len(bins)
    for b in bins[::-1]:
        k -= 1
        ans[column <= b] = k
    np.nan_to_num(ans, copy=False, nan=-1)
    return ans

def binner(df, columns, all_bins):
    assert len(columns) == len(all_bins)
    for column, bins in zip(columns, all_bins):
        df[column] = bin_column(df[column], bins)

#test
df = pd.read_csv('./data/train.csv', index_col='id')

binner(df, ['osm_finance_points_in_0.001', 'osm_finance_points_in_0.005'], [[0, 1, 2], [0, 1, 3, 5, 10]])

print(df['osm_finance_points_in_0.001'].head(100).to_numpy())