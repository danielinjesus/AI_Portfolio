# %%
import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('titanic.csv')

# 모든 행과 열을 표시하도록 설정
pd.set_option('display.max_rows', None)  # 모든 행 표시
pd.set_option('display.max_columns', None)  # 모든 열 표시

# 데이터 출력
display(df)  # Jupyter Notebook에서 전체 테이블을 표시
# %%
import pandas as pd
df = pd.read_csv('/data/ephemeral/home/train.csv')
nullcnt = df.isnull().sum()
notnullcnt = df.notnull().sum()
nullnot = pd.concat([nullcnt + notnullcnt], axis=0)
result_df = pd.DataFrame({
    'null_count': nullcnt,
    'notnull_count': notnullcnt,
    'full_number' : nullnot })
display(result_df)
# %%