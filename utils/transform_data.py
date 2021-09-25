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


def transform(df):
    feats=['osm_catering_points_in_0.001','osm_catering_points_in_0.005','osm_catering_points_in_0.0075','osm_catering_points_in_0.01','osm_shops_points_in_0.001','osm_shops_points_in_0.005','osm_shops_points_in_0.0075','osm_shops_points_in_0.01','osm_offices_points_in_0.001','osm_offices_points_in_0.005','osm_offices_points_in_0.0075','osm_offices_points_in_0.01','osm_finance_points_in_0.001','osm_finance_points_in_0.005','osm_finance_points_in_0.0075','osm_finance_points_in_0.01','osm_healthcare_points_in_0.005','osm_healthcare_points_in_0.0075','osm_healthcare_points_in_0.01','osm_leisure_points_in_0.005','osm_leisure_points_in_0.0075','osm_leisure_points_in_0.01','osm_historic_points_in_0.005','osm_historic_points_in_0.0075','osm_historic_points_in_0.01','osm_building_points_in_0.001','osm_building_points_in_0.005','osm_building_points_in_0.0075','osm_building_points_in_0.01','osm_hotels_points_in_0.005','osm_hotels_points_in_0.0075','osm_hotels_points_in_0.01','osm_culture_points_in_0.001','osm_culture_points_in_0.005','osm_culture_points_in_0.0075','osm_culture_points_in_0.01','osm_amenity_points_in_0.001','osm_amenity_points_in_0.005','osm_amenity_points_in_0.0075','osm_amenity_points_in_0.01','osm_train_stop_points_in_0.005','osm_train_stop_points_in_0.0075','osm_train_stop_points_in_0.01','osm_transport_stop_points_in_0.005','osm_transport_stop_points_in_0.0075','osm_transport_stop_points_in_0.01','osm_crossing_points_in_0.001','osm_crossing_points_in_0.005','osm_crossing_points_in_0.0075','osm_crossing_points_in_0.01']
    bins_all=[[],[],[],[],[0, 1, 2, 3, 4],[],[],[],[0, 1, 2],[],[],[],[0, 1, 2],[0, 1, 3, 5, 10],[],[],[],[],[],[],[],[],[],[],[],[2,4],[25, 60],[],[50,150],[],[],[],[0, 1],[0, 1],[0, 1, 2],[0, 1, 2, 3],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    binner(df, feats, bins_all)

    regions = df['region'].unique()
    reg_dict = dict(zip(regions, range(len(regions))))
    df['region'] = [reg_dict[r] for r in df['region']]
    df['street'] = [s if type(s) == float else float(s[1:]) for s in df['street']]

    df_moscow = df[df['region'] == reg_dict['Москва']]
    df_peter =  df[df['region'] == reg_dict['Санкт-Петербург']]
    df_other = df[(df['region'] != reg_dict['Москва']) & (df['region'] != reg_dict['Санкт-Петербург'])]

    return df_moscow, df_peter, df_other