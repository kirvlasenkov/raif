import pandas as pd
import numpy as np
import xgboost as xgb

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


THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1

def deviation_metric_one_sample(y_true, y_pred):
    """
    Реализация кастомной метрики для хакатона.

    :param y_true: float, реальная цена
    :param y_pred: float, предсказанная цена
    :return: float, значение метрики
    """
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD:
        return 0
    elif deviation <= - 4 * THRESHOLD:
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / THRESHOLD) + 1) ** 2
    elif deviation < 4 * THRESHOLD:
        return ((deviation / THRESHOLD) - 1) ** 2
    else:
        return 9

def deviation_metric_optimized(y_true, y_pred, weights=1):
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    answer = np.zeros_like(deviation)

    answer[np.abs(deviation) <= THRESHOLD]  = 0
    answer[deviation < -THRESHOLD]          = NEGATIVE_WEIGHT * ((deviation[deviation < -THRESHOLD] / THRESHOLD) + 1) ** 2
    answer[deviation > THRESHOLD]           = ((deviation[deviation > THRESHOLD] / THRESHOLD) - 1) ** 2
    answer[deviation <= - 4 * THRESHOLD]    = 9 * NEGATIVE_WEIGHT
    answer[deviation >= 4 * THRESHOLD]      = 9

    if type(weights) == np.ndarray:
        return (answer * weights).mean() / weights.mean() 
    else:
        return answer.mean()

def dev_metric_grad(pred, y):

    deviation = (pred - y) / np.maximum(1e-8, y)

    answer = np.zeros_like(deviation)

    answer[np.abs(deviation) <= THRESHOLD]  = 0
    answer[deviation < -THRESHOLD]          = NEGATIVE_WEIGHT * ((deviation[deviation < -THRESHOLD] / THRESHOLD) + 1) * 2
    answer[deviation > THRESHOLD]           = ((deviation[deviation > THRESHOLD] / THRESHOLD) - 1) * 2
    answer[deviation <= - 4 * THRESHOLD]    = 0
    answer[deviation >= 4 * THRESHOLD]      = 0

    return answer

def dev_metric_hess(pred, y):

    deviation = (pred - y) / np.maximum(1e-8, y)

    answer = np.zeros_like(deviation)

    answer[np.abs(deviation) <= THRESHOLD]  = 0
    answer[deviation < -THRESHOLD]          = NEGATIVE_WEIGHT / THRESHOLD * 2
    answer[deviation > THRESHOLD]           = 1 / THRESHOLD * 2
    answer[deviation <= - 4 * THRESHOLD]    = 0
    answer[deviation >= 4 * THRESHOLD]      = 0

    return answer

def dev_metric(pred, true):
    if type(true) == xgb.DMatrix:
        y = true.get_label()
    else:
        y = true
    return dev_metric_grad(pred, y), dev_metric_hess(pred, y)

def deviation_metric(y_true: np.array, y_pred: np.array) -> float:
    return np.array([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()


# feats=['osm_catering_points_in_0.001','osm_catering_points_in_0.005','osm_catering_points_in_0.0075','osm_catering_points_in_0.01','osm_shops_points_in_0.001','osm_shops_points_in_0.005','osm_shops_points_in_0.0075','osm_shops_points_in_0.01','osm_offices_points_in_0.001','osm_offices_points_in_0.005','osm_offices_points_in_0.0075','osm_offices_points_in_0.01','osm_finance_points_in_0.001','osm_finance_points_in_0.005','osm_finance_points_in_0.0075','osm_finance_points_in_0.01','osm_healthcare_points_in_0.005','osm_healthcare_points_in_0.0075','osm_healthcare_points_in_0.01','osm_leisure_points_in_0.005','osm_leisure_points_in_0.0075','osm_leisure_points_in_0.01','osm_historic_points_in_0.005','osm_historic_points_in_0.0075','osm_historic_points_in_0.01','osm_building_points_in_0.001','osm_building_points_in_0.005','osm_building_points_in_0.0075','osm_building_points_in_0.01','osm_hotels_points_in_0.005','osm_hotels_points_in_0.0075','osm_hotels_points_in_0.01','osm_culture_points_in_0.001','osm_culture_points_in_0.005','osm_culture_points_in_0.0075','osm_culture_points_in_0.01','osm_amenity_points_in_0.001','osm_amenity_points_in_0.005','osm_amenity_points_in_0.0075','osm_amenity_points_in_0.01','osm_train_stop_points_in_0.005','osm_train_stop_points_in_0.0075','osm_train_stop_points_in_0.01','osm_transport_stop_points_in_0.005','osm_transport_stop_points_in_0.0075','osm_transport_stop_points_in_0.01','osm_crossing_points_in_0.001','osm_crossing_points_in_0.005','osm_crossing_points_in_0.0075','osm_crossing_points_in_0.01']

# bins_all=[[],[],[],[],[0, 1, 2, 3, 4],[],[],[],[0, 1, 2],[],[],[],[0, 1, 2],[0, 1, 3, 5, 10],[],[],[],[],[],[],[],[],[],[],[],[2,4],[25, 60],[],[50,150],[],[],[],[0, 1],[0, 1],[0, 1, 2],[0, 1, 2, 3],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]


# binner(df, feats, bins_all)

# def calc_density(df):
#     for i in range

#test
# df = pd.read_csv('./data/train.csv', index_col='id')
# binner(df, ['osm_finance_points_in_0.001', 'osm_finance_points_in_0.005'], [[0, 1, 2], [0, 1, 3, 5, 10]])
# print(df['osm_finance_points_in_0.001'].head(100).to_numpy())