# %%
import os
# 현재 파일 이름을 확장자 포함해서 가져옴
current_file_name = os.path.basename(__file__)
print(current_file_name)

# 확장자를 제외한 파일 이름
current_file_name_without_extension = os.path.splitext(current_file_name)[0]
print(current_file_name_without_extension)

location = '/data/ephemeral/home/AI_Portfolio/AI_Projects/House_Price_Prediction/competition/submission/'
file_name = current_file_name_without_extension
location_file_name = location + file_name

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
test_path  = '/data/ephemeral/home/AI_Portfolio/AI_Projects/House_Price_Prediction/competition/data/test.csv'
dt_train = pd.read_csv(train_path)
dt_test = pd.read_csv(test_path)

# 9000건 Label 인코딩 실패이유. test랑 합쳐서 인코딩 안 해서.

dt_train['is_test'] = 0
dt_test['is_test'] = 1
concat = pd.concat([dt_train, dt_test]) # 하나의 데이터로 만들어줍니다.
print(concat.columns)
# %%

concat['full_addr'] = concat['시군구'] + concat['번지'].fillna('')
concat = concat[['전용면적(㎡)', '계약년월', '시군구', '층', 'full_addr', '건축년도', 'target', 'is_test']]
concat = concat.rename(columns={'전용면적(㎡)':'apt_area'})
concat['full_addr_encoded'] = LabelEncoder().fit_transform(concat['full_addr'])
concat = concat.drop(columns=['full_addr'])
print(concat.columns)
#%%

# print(concat['full_addr_encoded'].value_counts())
# 계약년과 계약월을 분리하여 컴터가 날짜로 더 잘 인지하도록 도와준다.
concat['apt_area'] = concat['apt_area'].round(2)

concat['cont_year'] = concat['계약년월'].astype('str').map(lambda x : x[:4])
concat['cont_month'] = concat['계약년월'].astype('str').map(lambda x : x[4:])
del concat['계약년월']
concat['cont_year'] = concat['cont_year'].astype(int)
concat['cont_month'] = concat['cont_month'].astype(int)

# 구와 동을 분리칼럼저장하고 One-hot encoding
concat['구'] = concat['시군구'].map(lambda x : x.split()[1])
concat['동'] = concat['시군구'].map(lambda x : x.split()[2])
concat = pd.get_dummies(concat, columns=['구', '동'])
del concat['시군구']

# 또한 신축인지, 구축인지의 여부도 실거래가에 큰 영향을 줄 수 있습니다.
# 따라서 건축년도에 따라 파생변수를 제작해주도록 하겠습니다.

# 건축년도 분포는 아래와 같습니다. 특히 2005년이 Q3에 해당합니다.
# 2009년 이후에 지어진 건물은 10%정도 되는 것을 확인할 수 있습니다.
concat['건축년도'].describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.8, 0.9])

# 따라서 2009년 이후에 지어졌으면 비교적 신축이라고 판단하고, 신축 여부 변수를 제작해보도록 하겠습니다.
concat['신축여부'] = concat['건축년도'].apply(lambda x: 1 if x >= 2009 else 0)
concat['신축여부'] = concat['건축년도'].apply(lambda x: 2 if x >= 1973 | x <= 1979 else 0)
concat.head(1)       # 최종 데이터셋은 아래와 같습니다.
concat.shape

print('===내용 보기===')
print(concat.info())
print(concat.columns)
print(concat.head(5))
print(concat.shape)
#%%

# 이제 다시 train과 test dataset을 분할해줍니다. 위에서 제작해 놓았던 is_test 칼럼을 이용합니다.
dt_train = concat.query('is_test==0')
dt_test = concat.query('is_test==1')

print(f'train column들 : {dt_train.columns}')
print(f'test column들 : {dt_test.columns}')
#%%
# 이제 is_test 칼럼은 drop해줍니다.
dt_train.drop(['is_test'], axis = 1, inplace=True)
dt_test.drop(['is_test'], axis = 1, inplace=True)

print(f'target 분리전: {dt_train.shape}, {dt_test.shape}')

# Target과 독립변수들을 분리해줍니다.
dt_test.drop(['target'], axis = 1, inplace=True)

X_train = dt_train.copy()
# print(f'타겟 분리후: {X_train.shape}, {dt_test.shape}')
#%%
#####################################################################################################

# Valid set을 test.csv 데이터와 동일한 조건하에 만들기 위하여 훈련기간과 validation기간을 분리하여 Validation set은 훈련하지 못하도록 설계

# X_train['계약년'].dtypes

print('처리전 X_train',X_train.shape)

X_train['cont_year'] = X_train['cont_year'].astype(str)
X_train['cont_month'] = X_train['cont_month'].astype(str)

# print("계약월",set(X_train['계약월']))

X_val = X_train[(X_train['cont_year'] == '2023')|(X_train['cont_month'] == '6')]
X_train = X_train[~((X_train['cont_year'] == '2023')|(X_train['cont_month'] == '6')) ]

print('처리후 X_train',X_train.shape)
print('처리후 X_Val',X_val.shape)

X_train['cont_year'] = X_train['cont_year'].astype(int)
X_train['cont_month'] = X_train['cont_month'].astype(int)

X_val['cont_year'] = X_val['cont_year'].astype(int)
X_val['cont_month'] = X_val['cont_month'].astype(int)

y_train = X_train['target']
X_train = X_train.drop(['target'], axis=1)

y_val = X_val['target']
X_val = X_val.drop(['target'], axis=1)

print("모델 학습을 시작합니다.")
# RandomForestRegressor를 이용해 회귀 모델을 적합시키겠습니다.
model = RandomForestRegressor(n_estimators=5, criterion='squared_error', random_state=1, n_jobs=-1)
model.fit(X_train, y_train)
print("모델학습완료")
pred = model.predict(X_val)

# 랜덤포레스트의 하이퍼파라미터도 데이터에 맞게 지정해줄 수 있습니다
# 데이터에 맞는 하이퍼파라미터를 찾는 것도 성능 향상에 도움이 될 수 있습니다.
# 회귀 관련 metric을 통해 train/valid의 모델 적합 결과를 관찰합니다.
print(f'RMSE test: {np.sqrt(metrics.mean_squared_error(y_val, pred))}')

#####################################################################################################
# RandomForestRegressor를 이용해 회귀 모델을 적합시키겠습니다.
X_train = dt_train.copy()

y_train = X_train['target']
X_train = X_train.drop(['target'], axis=1)

model = RandomForestRegressor(n_estimators=5, criterion='squared_error', random_state=1, n_jobs=-1)
model.fit(X_train, y_train)

pkl_location = '/data/ephemeral/home/model-pkl/'

# 학습된 모델을 저장합니다. Pickle 라이브러리를 이용하겠습니다.

pkl_location_file_name = pkl_location + file_name + '.pkl'
with open(f'{pkl_location_file_name}', 'wb') as f:
    pickle.dump(model, f)

# 저장된 모델을 불러옵니다.
with open(pkl_location_file_name, 'rb') as f:
    model = pickle.load(f)

X_test = dt_test

# Test dataset에 대한 inference를 진행합니다.
real_test_pred = model.predict(X_test)

output = location_file_name + ".csv"
  
# 앞서 예측한 예측값들을 저장합니다.
preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])
print(f"preds 모양 = {preds_df.shape}")

try:
    preds_df.to_csv(output, index=False)
    print("CSV 파일이 성공적으로 저장되었습니다.")
except Exception as e:
    print(f"파일 저장 중 오류 발생: {e}")