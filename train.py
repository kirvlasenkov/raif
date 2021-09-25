import pandas as pd
import numpy as np
import os, os.path
import re
import pickle

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostRegressor, CatBoostClassifier
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper, DeltaYStopper
from skopt import BayesSearchCV

text_floor_replace = {
                     'подвал, 1': 1, 
                     'подвал': -1, 
                     'цоколь, 1': 1, 
                     '1,2,антресоль': 2, 
                     'цоколь': 0, 
                     'тех.этаж (6)': 6, 
                     'Подвал': -1, 
                     'Цоколь': 0, 
                     'фактически на уровне 1 этажа': 1, 
                     '1,2,3': 3, 
                     '1, подвал': 1, 
                     '1,2,3,4': 4, 
                     '1,2': 2, 
                     '1,2,3,4,5': 5, 
                     '5, мансарда': 5, 
                     '1-й, подвал': 1, 
                     '1, подвал, антресоль': 1, 
                     'мезонин': 2, 
                     'подвал, 1-3': 3, 
                     '1 (Цокольный этаж)': 0, 
                     '3, Мансарда (4 эт)': 4, 
                     'подвал,1': 1, 
                     '1, антресоль': 1, 
                     '1-3': 3, 
                     'мансарда (4эт)': 4, 
                     '1, 2.': 2, 
                     'подвал , 1 ': 1, 
                     '1, 2': 2, 
                     'подвал, 1,2,3': 3, 
                     '1 + подвал (без отделки)': 1, 
                     'мансарда': 1, 
                     '2,3': 3, 
                     '4, 5': 5, 
                     '1-й, 2-й': 2, 
                     '1 этаж, подвал': 1, 
                     '1, цоколь': 1, 
                     'подвал, 1-7, техэтаж': 7, 
                     '3 (антресоль)': 3, 
                     '1, 2, 3': 3, 
                     'Цоколь, 1,2(мансарда)': 2, 
                     'подвал, 3. 4 этаж': 4, 
                     'подвал, 1-4 этаж': 4, 
                     'подва, 1.2 этаж': 4, 
                     '2, 3': 3, 
                     '7,8': 8, 
                     '1 этаж': 1, 
                     '1-й': 1, 
                     '3 этаж': 3, 
                     '4 этаж': 4, 
                     '5 этаж': 5, 
                     'подвал,1,2,3,4,5': 5, 
                     'подвал, цоколь, 1 этаж': 1, 
                     '3, мансарда': 3,
                    ' 1, 2, Антресоль': 2,
                    ' 1-2, подвальный': 2,
                    '1 (по док-м цоколь)': 1,
                    '1, 2 этаж': 2,
                    '1, 2, 3, мансардный': 3,
                    '1,2 ': 2,
                    '1,2,3 этаж, подвал': 3,
                    '1,2,3, антресоль, технический этаж': 3,
                    '1,2,3,4, подвал': 4,
                    '1,2,подвал ': 2,
                    '1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)': 3,
                    '1-7': 7,
                    '2, 3, 4, тех.этаж': 4,
                    '2-й': 2,
                    '3 этаж, мансарда (4 этаж)': 4,
                    '3, 4': 4,
                    '3,4': 4,
                    '5(мансарда)': 5,
                    'Техническое подполье': -1,
                    'подвал, 1 и 4 этаж': 4,
                    'подвал, 1, 2': 2,
                    'подвал, 1, 2, 3': 3,
                    'подвал, 2': 2,
                    'подвал,1,2,3': 3,
                    'технический этаж,5,6': 6,
                    'цоколь, 1, 2,3,4,5,6': 6,
                    'цокольный': 0,
                    'цокольный, 1,2':2
                    }
                    
features_for_log = [
                   'osm_city_closest_dist',
                    'osm_train_stop_closest_dist',
                    'osm_transport_stop_closest_dist',
                    'osm_crossing_closest_dist',
                    'reform_mean_floor_count_500'
                   ]
                   
drop_cols = [
            'date', 
            'city',
            'street',
            'clear_city',
            ]
            
            
cols_to_discretize = {
                    'osm_shops_points_in_0.001': 2,
                    'osm_shops_points_in_0.005': 2,
                    'osm_shops_points_in_0.0075': 2,
                    'osm_shops_points_in_0.01': 2,
                    'osm_finance_points_in_0.005': 2,
                    'osm_finance_points_in_0.0075': 2,
                    'osm_finance_points_in_0.01': 2,
                    'osm_leisure_points_in_0.005': 2,
                    'osm_leisure_points_in_0.01': 2,
                    'osm_amenity_points_in_0.01': 2,
                    'osm_transport_stop_points_in_0.005': 2,
                    'osm_crossing_points_in_0.005': 2,
                    'osm_crossing_points_in_0.0075': 2,
                    'osm_crossing_points_in_0.01': 2,
                    'reform_count_of_houses_1000': 2,
                    'reform_count_of_houses_500': 2,
                    'reform_mean_year_building_500': 2,
                    'reform_mean_year_building_1000': 2,
                    'osm_transport_stop_points_in_0.0075': 3,
                    'osm_transport_stop_points_in_0.01': 3
                }
                
not_inc_regions = [ 
                    'армавир',
                    'амурская область',
                    'архангельсая область без ао',
                    'архангельская область',
                    'архангельская область без авт. округа',
                    'архангельская область без авт.округа',
                    'архангельская область без автономного округа',
                    'архангельская область без ао',
                    'архангельская область без нао',
                    'архангельская область кроме ненецкого автономного округа',
                    'астраханская область',
                    'белская область',
                    'в том числе: ненецкий автономный округ',
                    'владимирская область',
                    'вологда',
                    'дагестан',
                    'дальневосточный',
                    'еврейская авт.область',
                    'еврейская автономная область',
                    'забайкальский край',
                    'ингушетия',
                    'кабардино-балкарская',
                    'калмыкия',
                    'камчатский край',
                    'карачаево-черкесская',
                    'краснодар',
                    'крым',
                    'курганская область',
                    'магаданская область',
                    'марий эл',
                    'мурманская область',
                    'ненецкий авт.округ',
                    'ненецкий автономный округ',
                    'ненецкий автономный округ архангельская область',
                    'ненецкий ао',
                    'новороссийск',
                    'новская область',
                    'оренбургская область',
                    'отгружено товаров собственного производства, выполнено работ и услуг собственными силами малыми предприятиями включая микропредприятия',
                    'приволжский',
                    'псковская область',
                    'рязанская область',
                    'саха якутия',
                    'сахалинская область',
                    'севастополь',
                    'северная осетия - алания',
                    'северная осетия- алания',
                    'северная осетия-алания',
                    'северо-западный',
                    'северо-кавказский',
                    'сибирский',
                    'сочи',
                    'средняя численность работников малых предприятий без микропредприятий',
                    'средняя численность работников микропредприятий',
                    'тамбовская область',
                    'тверская область',
                    'томск',
                    'уральский',
                    'хабаровский край',
                    'хакасия',
                    'череповец',
                    'чеченская',
                    'чита',
                    'чувашская',
                    'чувашская - чувашия',
                    'чувашская -чувашия',
                    'чукотский авт.округ',
                    'чукотский автономный округ',
                    'южный',
                    'ямало-ненецкий авт.округ',
                    'ямало-ненецкий автономный округ',
                    'ямало-ненецкий автономный округ тюменская область',
                    'ямало-ненецкий ао',
                    'nan'
]

def is_float(value):
    try:
        if type(float(value)) != 'float':
            return True
    except:
        return False
        
def create_agg_osm_features(train_data, test_data=None):
    osm_cols = [col for col in list(train_data) if 'osm_' in col and 'name' not in col]
    
    for col in osm_cols:
        train_data[col] = train_data[col].astype('float64')
        agg_data = (train_data.groupby('clear_city')[col].sum() / train_data[col].sum() * 100).reset_index()
        if type(test_data) != type(None):
            test_data = pd.merge(test_data, agg_data, on='clear_city', how='left', suffixes=('', '_agg')) 
        train_data = pd.merge(train_data, agg_data, on='clear_city', how='left', suffixes=('', '_agg'))
        
    if type(test_data) != type(None):
        return train_data, test_data
    else:
        return train_data

def log_features(data, lst):
    for col in lst:
        name_log_col = '_'.join(['log', col])
        data[name_log_col] = np.log(data[col] + 0.00001)
    return data

def filter_outliers(data, cols_to_filter: list):

    for col in cols_to_filter:
        data = data[(data[col] < data[col].mean() + 3 * data[col].std()) & (data[col] > data[col].mean() - 3 * data[col].std())]
    
    return data
    
def bins_discretize_data(train_data, test_data, columns):
    data = pd.concat([train_data, test_data])
    for col, n_bins in columns.items():
        if data[col].isnull().sum() == 0:
            est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
            discr_col = est.fit_transform(np.array(data[col]).reshape(-1, 1))
            name_col = 'bin_' + col
            data[name_col] = discr_col
    return data
    
def replaces(data):
    map_repl = {'город федерального значения': '',
                'город': '',
                ')': '',
                '(': '',
                'h': 'н',
                'k': 'к',
                'республика': '',
                'г.': '',
               'столица российской федерации': ''}
    for old_value, new_value in map_repl.items():
        data['region'] = data['region'].apply(lambda x: str(x).lower().replace(old_value, new_value).strip())
    data['region'] = data['region'].apply(lambda x: re.sub(r'\s+', ' ', x))
    return data
    
    
def preprocess_data(data):
    for col in data.columns[1:]:
        data = replaces(data)
        data[col] = data[col].apply(lambda x: str(x).strip().replace('-', 'nan').replace(' ', '').replace(',', '.')).astype('float')
        data = data[data['region'].apply(lambda x: 'федерация' not in x and 'федеральный округ' not in x)]
        data = data.iloc[:-1]

        regions_for_repl = {
            'тюменская область без ао': 'тюменская область',
            'удмуртская': 'удмуртия',
            'ханты-мансийский ао-югра': 'ханты-мансийский ао',
            'кемеровская область - кузбасс': 'кемеровская область',
            'адыгея адыгея': 'адыгея',
            'тюменская область без автономных округов': 'тюменская область',
            'тюменская область без авт. округов': 'тюменская область',
            'белская область': 'белгородская область',
            'нижеская область': 'нижегородская область',
            'татарстан татарстан': 'татарстан',
            'тюменская область без авт.округов': 'тюменская область'
        }
        
        data.loc[data['region'].apply(lambda x: 'ханты' in x), 'region'] = 'ханты-мансийский ао'
        
        for old_region_name, new_region_name in regions_for_repl.items():
            data['region'] = data['region'].apply(lambda x: x.replace(old_region_name, new_region_name))
        
    return data
    
    
def merge_data(file_paths):
    for i, path in enumerate(file_paths):

        data = pd.read_csv(path, sep='\t', encoding='windows-1251')
        data = preprocess_data(data)
        
        if i == 0:
            merged_data = data.copy()
        elif i != 0:
            merged_data = pd.merge(merged_data, data, how='outer', on='region')
            
    cols = [col for col in merged_data.columns if 'Unnamed' not in col]
    
    return merged_data[cols]
    
#create add_data part

file_paths = [os.path.join(os.getcwd(), 'rosstat', 'csv', file_name) for file_name in os.listdir(os.path.join(os.getcwd(), 'rosstat', 'csv'))]

merged_data = merge_data(file_paths)

for region in not_inc_regions:
    merged_data = merged_data[merged_data['region'] != region].copy()
    
merged_data.drop_duplicates('region', keep='last', inplace=True)

cols_with_nulls = merged_data.isnull().sum()
cols_with_nulls = cols_with_nulls[cols_with_nulls > 0].index

for col in cols_with_nulls:
    merged_data.fillna(merged_data[col].median(), inplace=True)
    
merged_data.reset_index(inplace=True, drop=True)

merged_data.to_csv('./data/data/add_data.csv', index=False)

#feature_engineering part
                    
train_data = pd.read_csv('./data/data/train.csv')
test_data = pd.read_csv('./data/data/test.csv')

train_data['day'] = pd.to_datetime(train_data['date']).dt.day
test_data['day'] = pd.to_datetime(test_data['date']).dt.day

train_data['floor_isnull'] = np.where(train_data['floor'].isnull(), 0, 1)
test_data['floor_isnull'] = np.where(test_data['floor'].isnull(), 0, 1)

train_data['floor_is_float'] = np.where(train_data['floor'].apply(lambda x: is_float(x)), 1, 0)
test_data['floor_is_float'] = np.where(test_data['floor'].apply(lambda x: is_float(x)), 1, 0)

train_data['address'] = train_data['city'] + ' ' + train_data['street']
test_data['address'] = test_data['city'] + ' ' + test_data['street']

train_data['log_total_square'] = np.log(train_data['total_square'])
test_data['log_total_square'] = np.log(test_data['total_square'])
        
train_data['clear_city'] = train_data['city'].apply(lambda x: x.split(', ')[-1].strip())
test_data['clear_city'] = test_data['city'].apply(lambda x: x.split(', ')[-1].strip())

train_data, test_data = create_agg_osm_features(train_data, test_data)

add_data = pd.read_csv('./data/data/add_data.csv')

train_data['region'] = train_data['region'].apply(lambda x: x.lower().strip())
test_data['region'] = test_data['region'].apply(lambda x: x.lower().strip())

train_data = pd.merge(train_data, add_data, on='region', how='left', suffixes=('', ''))
test_data = pd.merge(test_data, add_data, on='region', how='left', suffixes=('', ''))

train_data['text_floor'] = np.where(train_data['floor'].isnull(), 
                                    'unknown_floor', 
                                    train_data['floor'].apply(lambda x: str(x)))

train_data.loc[train_data['floor_is_float'] == 0, 'floor'] = \
    train_data.loc[train_data['floor_is_float'] == 0, 'floor'].map(text_floor_replace)

test_data['text_floor'] = np.where(test_data['floor'].isnull(), 
                                    'unknown_floor', 
                                    test_data['floor'].apply(lambda x: str(x)))

test_data.loc[test_data['floor_is_float'] == 0, 'floor'] = \
    test_data.loc[test_data['floor_is_float'] == 0, 'floor'].map(text_floor_replace)
    
        
train_data = log_features(train_data, features_for_log)
test_data = log_features(test_data, features_for_log)

train_data = filter_outliers(train_data, ['reform_mean_year_building_1000', 'reform_mean_year_building_500'])

eur_rub = pd.read_csv('eur_rub.txt', sep=',')
usd_rub = pd.read_csv('usd_rub.txt', sep=',')

train_data = pd.merge(train_data, eur_rub, on='date', how='left')
test_data = pd.merge(test_data, eur_rub, on='date', how='left')

train_data.drop(drop_cols, axis=1, inplace=True)
test_data.drop(drop_cols, axis=1, inplace=True)

data = bins_discretize_data(train_data, test_data, cols_to_discretize)
train_data = data[~data['per_square_meter_price'].isnull()]
test_data = data[data['per_square_meter_price'].isnull()]
test_data['address'].fillna(method='backfill', inplace=True)

#train_part

train_data = train_data[train_data['price_type'] == 1]
kf = KFold(3)

cols = [col for col in list(train_data) if col != 'id' not in col and 'per_square_meter_price' not in col and 'price_type' not in col]

catboost_params = {
                    'iterations': np.arange(100, 1100, 100),
                    'depth': np.arange(6, 16, 2),
                    'learning_rate': np.arange(0.01, 1, 0.02),
                    'random_strength': np.logspace(-9, 0, 10),
                    'bagging_temperature': np.arange(0.1, 2.1, 0.1),
                    'border_count': np.arange(1, 255, 1),
                    'l2_leaf_reg': np.arange(1, 19, 1),
                    'min_data_in_leaf': np.arange(1, 7, 1)
}

catboost = CatBoostRegressor(
    loss_function='RMSE',
    eval_metric='R2',
    task_type='CPU',
    cat_features=['osm_city_nearest_name', 
                  'day', 
                  'realty_type', 
                  'region',
                  'address',
                  'text_floor']
)

opt_catboost = BayesSearchCV(catboost, catboost_params, cv=kf, n_iter=100, scoring='r2', random_state=42)
opt_catboost.fit(train_data[cols], train_data['per_square_meter_price'], callback=[VerboseCallback(100), DeadlineStopper(60*300), DeltaYStopper(0.001, 5)])

output_regressor = open('./models/regressor.pkl', 'wb')
pickle.dump(opt_catboost, output_regressor)

test_pred = opt_catboost.predict(test_data[cols])
pd.DataFrame(np.c_[test_data['id'], test_pred], columns=['id', 'per_square_meter_price']).set_index('id').to_csv('test_submission.csv')