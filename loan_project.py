import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from scipy.interpolate import interp1d
import matplotlib.font_manager as fm

# 폰트 설정
font_path = 'C:/Windows/Fonts/NanumGothic.ttf'  # 폰트 경로 (환경에 따라 변경)
font_prop = fm.FontProperties(fname=font_path, size=12)
plt.rc('font', family=font_prop.get_name())

# 이미지 저장할 디렉토리 생성
output_dir = 'image'
os.makedirs(output_dir, exist_ok=True)

# 데이터 수집 탐색
df = pd.read_csv("LoanDataset.csv")  # CSV 파일에서 데이터프레임으로 데이터 수집
engine = create_engine('mysql+pymysql://root:1234@localhost/loan_dataset')

# 데이터프레임을 SQL 테이블에 저장
df.to_sql('loan_data', con=engine, if_exists='append', index=False)

# 데이터베이스 연결 및 데이터 불러오기
conn = engine.connect()
query = "SELECT * FROM loan_data;"
df = pd.read_sql_query(query, conn)
conn.close()  # 연결 종료

# 데이터 정보 확인
df.info()
df.head()
df.isnull().sum()
pd.set_option('display.max_columns', None)  # 모든 열 표시
pd.set_option('display.max_rows', None)     # 모든 행 표시

# 결측값 처리 함수 정의
def fill_missing_values(df):
    df['historical_default'] = df['historical_default'].fillna('Not_available')
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
    df['employment_duration'] = df['employment_duration'].fillna(df['employment_duration'].mean())
    df['Current_loan_status'] = df['Current_loan_status'].fillna(df['Current_loan_status'].mode()[0])
    return df

# 데이터 정제
df = fill_missing_values(df)

# customer_id 컬럼 삭제
df.drop('customer_id', axis=1, inplace=True)

# 데이터 변환
# Current_loan_status 변수 라벨 인코딩
label_encoder = LabelEncoder()
df['Current_loan_status'] = label_encoder.fit_transform(df['Current_loan_status'])
print(df['Current_loan_status'].unique())

# customer_income 데이터 변환 함수 정의
def convert_income(df):
    df['customer_income'] = df['customer_income'].str.replace(',', '', regex=True)
    df['customer_income'] = pd.to_numeric(df['customer_income'])
    return df

df = convert_income(df)

# 데이터 요약 통계 확인
summary = df.describe()
print(summary)

# 입력 변수와 타겟 변수 분리
numdf = df.select_dtypes(exclude=object)  # 숫자형 데이터
catdf = df.select_dtypes(include=object)   # 범주형 데이터

target = numdf['Current_loan_status']
features = numdf.drop(columns=['Current_loan_status'])

# target을 2차원 배열로 변환
target = target.values.reshape(-1, 1)

# 데이터 스케일링(표준화)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 히트맵 상관계수 표시
plt.figure(figsize=(7, 6))
sns.heatmap(numdf.corr(), cmap='coolwarm', annot=True)
plt.savefig(os.path.join(output_dir, '상관계수 히트맵'), bbox_inches='tight')

# 데이터 차원 확인
print(numdf.shape)  # 숫자형 데이터 열 수
print(catdf.shape)  # 범주형 데이터 열 수

# 대출 금액 분포 히스토그램 함수 정의
def plot_histogram(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 6))
    sns.histplot(data, bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(output_dir, filename))

# 대출 금액 분포 히스토그램
plot_histogram(np.log1p(df['loan_amnt']), '대출 금액 (로그 변환) 분포 히스토그램', 
                '로그 변환된 대출 금액', '빈도', '대출 금액 히스토그램')

# 이상치 제거
filtered_df = df[df['loan_amnt'] < df['loan_amnt'].quantile(0.99)]

# 이상치 제거 히스토그램
plot_histogram(filtered_df['loan_amnt'], '대출 금액 (이상치 제거) 분포 히스토그램', 
                '대출 금액', '빈도', '대출 금액 히스토그램_이상치제거')

# 이자율과 상환 상태 간의 관계
# 이상치 인덱스 찾기
outliers = df[(df['customer_age'] < 20) | (df['customer_age'] > 80)].index

# 이상치 제거
df_cleaned = df.drop(outliers)

# 산점도 시각화
plt.figure(figsize=(12, 10))
sns.scatterplot(data=df_cleaned, x='customer_age', y='cred_hist_length', hue='Current_loan_status', alpha=0.7)
plt.xlabel('고객 연령')
plt.ylabel('신용 기록 기간')
plt.title('고객 연령과 신용 기록 기간의 관계')
plt.legend(title='현재 대출 상태')
plt.savefig(os.path.join(output_dir, '고객 연령과 신용 기록 기간의 관계'))

# 범주형 변수의 분포
plt.figure(figsize=(10, 6))
sns.countplot(x='Current_loan_status', data=df, color='orange')
plt.xlabel('대출 상태')
plt.ylabel('대출 건수')
plt.title('대출 상태 분포')
plt.savefig(os.path.join(output_dir, '대출 상태 분포'))

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# 모델 초기화
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

# 성능 지표 저장
results = {}

# 모델 학습 및 평가
for model_name, model in models.items():
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)
    
    results[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

# 결과 출력
results_df = pd.DataFrame(results).T
print(results_df)

# 하이퍼 파라미터 튜닝 함수 정의
def tune_model(model, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=1)
    grid_search.fit(X_train, y_train.ravel())
    return grid_search.best_estimator_

# 각 모델의 하이퍼파라미터 튜닝
best_rf_model = tune_model(RandomForestClassifier(), {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
})
best_xgb_model = tune_model(XGBClassifier(eval_metric='mlogloss'), {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
})
best_lr_model = tune_model(LogisticRegression(max_iter=200), {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
})

# 튜닝된 모델 성능 평가
tuned_models = {
    'Logistic Regression': best_lr_model,
    'Random Forest': best_rf_model,
    'XGBoost': best_xgb_model
}

# 성능 지표 저장 및 예측 확률 저장
tuned_results = {}
probabilities = {}

# 모델 학습 및 평가
for model_name, model in tuned_models.items():
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    # 예측 확률 계산
    prob = model.predict_proba(X_test)[:, 1]  # 불이행 확률
    probabilities[model_name] = prob
    
    tuned_results[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

# 결과 출력
tuned_results_df = pd.DataFrame(tuned_results).T
print(tuned_results_df)

# 예측 확률 출력
for model_name, prob in probabilities.items():
    print(f"{model_name} 불이행 확률:", prob)

# 변수 중요도 시각화 (랜덤 포레스트)
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("변수 중요도 시각화(랜덤 포레스트)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), features.columns[indices], rotation=10)
plt.xlim([-1, X_train.shape[1]])
plt.ylabel('중요도')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '변수 중요도 시각화'))

# 모델 저장
joblib.dump(best_rf_model, 'best_rf_model.pkl')
joblib.dump(best_lr_model, 'best_lr_model.pkl')
joblib.dump(best_xgb_model, 'best_xgb_model.pkl')

# ROC 곡선 플롯
plt.figure(figsize=(10, 6))

# 각 모델에 대해 ROC 곡선 그리기
for model_name, model in tuned_models.items():
    print(f"Fitting model: {model_name}")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]  # 긍정 클래스 확률
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # 부드러운 곡선 생성
    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    unique_tpr = tpr[unique_indices]
    f = interp1d(unique_fpr, unique_tpr, kind='cubic', fill_value="extrapolate")
    fpr_new = np.linspace(0, 1, 200)  # 200개의 점으로 보간
    tpr_new = f(fpr_new)
    
    plt.plot(fpr_new, tpr_new, label=f'{model_name} (AUC = {roc_auc:.2f})')

# ROC 곡선 세팅
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('거짓 긍정 비율 (FPR)')
plt.ylabel('진짜 긍정 비율 (TPR)')
plt.title('수신자 조작 특성 곡선 (ROC 곡선)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve'))