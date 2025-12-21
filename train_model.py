import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\REINALDO\OneDrive\Escritorio\proyecto\insurance.csv')

df.head(3)
df.tail()

df.info()

df.shape

pd.set_option('display.float_format', '{:.2f}'.format)
df

df.duplicated().sum()

df.isnull().sum()

df.isna().sum().sum()

df.dropna(inplace=True)

df.shape

df.describe()

df.describe(include='all')

numeric_cols = ['age', 'bmi', 'bloodpressure', 'children','claim']
df[numeric_cols].hist(bins=20, figsize=(15, 10), color = 'skyblue', edgecolor = 'black')
plt.suptitle('Distribution', fontsize=16)

plt.show()

cat_cols = ['gender', 'diabetic','smoker', 'region']
plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_cols):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=col, data=df)
    plt.title(f'Count Plot of {col}')

    plt.tight_layout()
plt.show()

df.groupby(['gender', 'smoker'])['claim'].mean().round(2)

plt.figure(figsize=(8, 5))
sns.barplot(x='gender', y='claim', data=df, hue='smoker', estimator='mean', errorbar='sd')
plt.title('AVERAJE INSURANCE')

plt.show()

pivot_region_diabetic = df.groupby(['region', 'diabetic'])['claim'].mean().round(2).unstack()
pivot_region_diabetic

pivot_region_diabetic.plot(kind='bar', figsize=(8,5))
plt.title('Average claim by region & Diabetic Status')

plt.ylabel('Mean Claim')


plt.show()

pivot_region_diabetic = pd.pivot_table(df, values='claim', index='region', columns='smoker', aggfunc='mean')
pivot_region_diabetic

pivot_region_diabetic = pd.pivot_table(df, values='claim', index='children', columns='diabetic', aggfunc='mean')
pivot_region_diabetic

plt.figure(figsize=(8, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')

plt.show()

sns.scatterplot(x='age', y='claim', data=df, hue='smoker', style='gender', alpha = 0.7)
plt.title('Claim & Age By Smoker & Gender')

plt.show()

sns.regplot(data= df, x='bmi', y='claim', scatter_kws={'alpha':0.6})
plt.title('Claim & BMI')

plt.show()

sns.boxplot(data = df, x='children', y='claim')
plt.title('Claim & Children')

plt.show()

df['age_group'] = pd.cut(df['age'], bins=[0,18,30,45,60,100], labels=['0-18','19-30','31-45','46-60','60+'])
df

df['age_group'].value_counts()

sns.barplot(x='age_group', y='claim', data=df, estimator = 'mean', errorbar ='sd')
plt.title('Claim & Age Group')

plt.show()

df['bmi_category']= pd.cut(df['bmi'], bins=[0,18.5,24.9,29.9,100], labels=['Underweight','Normal','Overweight','Obesity'])
df

df['bmi_category'].value_counts().plot(kind='bar')
plt.title('BMI Category')

plt.show()

import warnings
warnings.filterwarnings('ignore')

sns.boxplot(data=df, x='bmi_category', y='claim', )
plt.title('Claim Distribution by BMI Category & smoking Status')

plt.show()

region_stats = df.groupby('region').agg(
    smoker_rate = ('smoker', lambda x: (x == 'Yes').mean()*100),
    mean_claim = ('claim', 'mean')

).reset_index()

region_stats

fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x='region', y='smoker_rate', data=region_stats, ax=ax1, alpha = 0.7)
ax2 = ax1.twinx()
sns.lineplot(x='region', y='mean_claim', data=region_stats, ax=ax2, color='red', marker='o')

ax1.set_ylabel('Smoker Rate (%)')
ax2.set_ylabel('Average Claim($)')
ax1.set_title('Smoker Rate and Average Claim by Region')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import joblib 

df.columns

x = df[['age', 'gender', 'bmi', 'bloodpressure', 'diabetic', 'children','smoker']]
y = df['claim']

x

cat_cols = ['gender', 'diabetic', 'smoker']
label_encoder = {}

for col in cat_cols:
  le = LabelEncoder()
  x[col] = le.fit_transform(x[col])
  label_encoder[col] = le

  joblib.dump(le, f'label_encoder_{col}.pkl')

x

label_encoder

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_cols = ['age', 'bmi', 'bloodpressure', 'children']
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

joblib.dump(scaler, 'scaler.pkl')

print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse}

results = {}

lr = LinearRegression()
lr.fit(X_train, y_train)
results['Linear Regression'] = evaluate_model(lr, X_train, X_test, y_train, y_test)
print('Linear Regression model trained')

best_poly_model = None
best_poly_score = -np.inf

for degree in [2,3]:
  poly = PolynomialFeatures(degree=degree)
  X_train_poly = poly.fit_transform(X_train)
  X_test_poly = poly.transform(X_test)

  poly_lr = LinearRegression()
  poly_lr.fit(X_train_poly, y_train)

  score = poly_lr.score(X_test_poly, y_test)

  if score > best_poly_score:
    best_poly_score = score
    best_poly_model = (degree, poly, poly_lr)

degree, poly, poly_lr = best_poly_model
results[f'Polynomial Regression (deg {degree})'] = evaluate_model(poly_lr, poly.fit_transform(X_train), poly.transform(X_test), y_train, y_test)
print('Polynomial Regression model are trained')

rf = RandomForestRegressor()

rf_params = {
    'n_estimators': [100,200],
    'max_depth': [None,10,20],
    'min_samples_split': [2,5],
    'min_samples_leaf': [1,2]
}

rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

results['Random Forest'] = evaluate_model(best_rf, X_train, X_test, y_train, y_test)
print('Random Forest model trained, best parameters', rf_grid.best_params_)

svr = SVR()

svr_params = {
    'kernel': ['rbf','poly','linear'],
    'C': [0.1, 10, 50],
    'epsilon': [0.1, 0.2, 0.5],
    'degree': [2,3]
}

svr_grid = GridSearchCV(svr, svr_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)

svr_grid.fit(X_train, y_train)

best_svr = svr_grid.best_estimator_

results['SVR'] = evaluate_model(best_svr, X_train, X_test, y_train, y_test)
print('SVR model trained, best parameters', svr_grid.best_params_)

xgb = XGBRegressor(objective='reg:squarederror')

xgb_params = {
    'n_estimators': [100,200],
    'max_depth': [3,5,7],
    'learning_rate': [0.01,0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]

}

xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

results['XGBoost'] = evaluate_model(best_xgb, X_train, X_test, y_train, y_test)

print('XGBoost model trained, best parameters', xgb_grid.best_params_)

results

results_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
results_df

best_rf

models = {
    'linear regression': lr,
    'polynomial regression': poly_lr,
    'random forest': best_rf,
    'svr': best_svr,
    'XGBoost': best_xgb
}

best_r2 = results_df['R2'].max()

best_r2

top_model = results_df[results_df['R2'] == best_r2]



top_model

best_model = models[top_model.index[0]]

best_model

joblib.dump(best_model, 'best_model.pkl')