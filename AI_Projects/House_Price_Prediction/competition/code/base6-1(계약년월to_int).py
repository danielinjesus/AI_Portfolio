# %%
location = '/data/ephemeral/home/model-pkl/'
file_name = '6_onehot_address_base'

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunGothic')                        # 이 폰트의 원하는 이름 설정
fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'}) # 폰트 설정
plt.rc('font', family='NanumBarunGothic')
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import warnings;warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import eli5
from eli5.sklearn import PermutationImportance
train_path = '/data/ephemeral/home/train.csv' 
test_path  = '/data/ephemeral/home/AI_Portfolio/AI_Projects/' \
                'House_Price_Prediction/competition/data/test.csv'
dt_train = pd.read_csv(train_path)
dt_test = pd.read_csv(test_path)
dt_train = dt_train[['전용면적(㎡)', '계약년월', '시군구', 'target']]
dt_test = dt_test[['전용면적(㎡)', '계약년월', '시군구']]

dt_train = dt_train.rename(columns={'전용면적(㎡)':'apt_area'})
dt_test = dt_test.rename(columns={'전용면적(㎡)':'apt_area'})

dt_test['apt_area'] = dt_test['apt_area'].round(2)

dt_train['cont_year'] = dt_train['계약년월'].astype('str').map(lambda x : x[:4])
dt_train['cont_month'] = dt_train['계약년월'].astype('str').map(lambda x : x[4:])
# dt_train['cont_date'] = pd.to_datetime(dt_train['cont_year']+'-'+dt_train['cont_month']+'-01')
del dt_train['계약년월']

dt_test['cont_year'] = dt_test['계약년월'].astype('str').map(lambda x : x[:4])
dt_test['cont_month'] = dt_test['계약년월'].astype('str').map(lambda x : x[4:])
# dt_test['cont_date'] = pd.to_datetime(dt_test['cont_year']+'-'+dt_test['cont_month']+'-01')
del dt_test['계약년월']

dt_train['cont_year'] = dt_train['cont_year'].astype(int)
dt_train['cont_month'] = dt_train['cont_month'].astype(int)

dt_test['cont_year'] = dt_test['cont_year'].astype(int)
dt_test['cont_month'] = dt_test['cont_month'].astype(int)

# 시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.
dt_train['구'] = dt_train['시군구'].map(lambda x : x.split()[1])
dt_train['동'] = dt_train['시군구'].map(lambda x : x.split()[2])
dt_train = pd.get_dummies(dt_train, columns=['구', '동'])
del dt_train['시군구']

dt_test['구'] = dt_test['시군구'].map(lambda x : x.split()[1])
dt_test['동'] = dt_test['시군구'].map(lambda x : x.split()[2])
dt_test = pd.get_dummies(dt_test, columns=['구', '동'])
del dt_test['시군구']

# Train과 Test의 원-핫 인코딩된 열을 일치시킴
dt_test = dt_test.reindex(columns=dt_train.columns, fill_value=0)
dt_test = dt_test.drop(columns=['target'], errors='ignore')

# print(pd.unique(dt_train['구']))
print(dt_train.info())
print(dt_train.columns)
# print(dt_test.info())
# %%
# print(dt_test.columns)
print(dt_train.head(5))
print(dt_test.head(5))
print(dt_train.shape, dt_test.shape)
#%%
# Target과 독립변수들을 분리해줍니다.
y_train = dt_train['target']
X_train = dt_train.drop(['target'], axis=1)

# RandomForestRegressor를 이용해 회귀 모델을 적합시키겠습니다.
model = RandomForestRegressor(n_estimators=5, criterion='squared_error', random_state=1, n_jobs=-1)
model.fit(X_train, y_train)

# 학습된 모델을 저장합니다. Pickle 라이브러리를 이용하겠습니다.
location_file_name = location + file_name + '.pkl'
with open(f'{location_file_name}', 'wb') as f:
    pickle.dump(model, f)

# 저장된 모델을 불러옵니다.
with open(location_file_name, 'rb') as f:
    model = pickle.load(f)

X_test = dt_test

# Test dataset에 대한 inference를 진행합니다.
real_test_pred = model.predict(X_test)

output = '/data/ephemeral/home/AI_Portfolio/AI_Projects/House_' \
         'Price_Prediction/competition/submission/' + file_name + 'output.csv'

# 앞서 예측한 예측값들을 저장합니다.
preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])
preds_df.to_csv(output, index=False)