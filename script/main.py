'''
House Price Prediction Model using XGBoost Pipelines
'''

#Importing all the packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


#Loading dataset
print("STEP 1: LOADING DATA")
print("-"*20)

data = pd.read_csv("dataset/housing.csv")

print(f"\n Data loaded successfully")
print(f"\n First 5 rows of data: ")
print(data.head())

print(f"\n Missing values in each column: ")
print(data.isnull().sum())


#Cleaning the dataset

print("STEP 2: CLEANING DATA")
print("-"*20)

if data['total_bedrooms'].isnull().sum() > 0:
    most_common_value = data['total_bedrooms'].mode()[0]
    data['total_bedrooms'].fillna(most_common_value, inplace=True)

print(f"\n Filled missing bedroom values with: {most_common_value}")


#Encoding the texts

print("STEP 3: CONVERTING TEXT TO NUMBERS")
print("-"*35)

print("\n Before encoding:")
print(data['ocean_proximity'].value_counts())

ocean_dummies = pd.get_dummies(data['ocean_proximity'], prefix= 'ocean')

data = pd.concat([data, ocean_dummies],axis=1)
data = data.drop('ocean_proximity', axis=1)

print(f"\n Converted text to {len(ocean_dummies.columns)} number columns")


#Creating visuals for checking important features

print("STEP 4: CREATING VISUALIZATIONS")
print("-"*31)

plt.figure(figsize=(14,5))


plt.subplot(1,2,1)
correlations = data.corr()['median_house_value'].sort_values(ascending=False)
correlations = correlations[1:]
plt.barh(range(len(correlations)), correlations.values)
plt.yticks(range(len(correlations)), correlations.index)
plt.xlabel('Correlations with House Price')
plt.title('Which features predict price best?')
plt.tight_layout()

plt.subplot(1,2,2)
plt.scatter(data['longitude'], data['latitude'], c = data['median_house_value'], cmap='viridis', alpha=0.4, s=10)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('House Price Across California')
plt.tight_layout()

plt.savefig('analysis.png', dpi = 300, bbox_inches = 'tight')
print("\n Charts saved as 'analysis.png'")
plt.show()


#Spliting the datset for further development

print("STEP 5: PREPARING DATA FOR ML")
print("-"*29)

X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

print(f"\n Features (X): {X.shape[1]} columns")
print(f"\n Target (y): house prices")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled (normalized)")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f" Data split complete: ")
print(f" Training set: {len(X_train)} houses")
print(f" Testing set: {len(X_test)} houses")


#Pre accuracy checking

print("STEP 6: TESTING MODEL RELIABILITY")
print("-"*33)

model = XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 6,
    random_state = 42,
    n_jobs = -1
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n Testing model accuracy")
r2_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')

print("\n Results from each fold:")
for i, score in enumerate(r2_scores, 1):
    print(f"Fold {i}: {score:.4f}({score*100:.2f}%)")

print(f"\n Average Score: {r2_scores.mean():.4f} ({r2_scores.mean()*100:.2f}%)")
print(f"Consistency : {r2_scores.std():.4f}")



#Model predictions

print("STEP 7: TRAINING FINAL MODEL")
print("-"*28)

print("\n Training XGBoost model on training data...")

final_model = XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 6,
    random_state = 42,
    n_jobs = -1
)
final_model.fit(X_train, y_train)
print("Model trained")

print("\n Making predictions on test data")
predictions = final_model.predict(X_test)
print("Predictions Complete")



#Checking the model accuracy

print("STEP 8: CHECKING MODEL ACCURACY")
print("-"*30)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("\n MODEL PERFORMANCE:")
print("-"*19)
print(f"R² Score:  {r2:.4f} ({r2*100:.2f}%)")
print(f"Model explain {r2*100:.1f}% of price variation")
print()
print(f"Average Error: ${mae:,.2f}")
print(f"On average, predictions are off by ${rmse:,.0f}")
print()
print(f"RMSE: ${mae:,.2f}")
print(f"Typical prediction error is ${rmse:,.0f}")


#Checking the important features

print("STEP 9: WHICH FEATURES MATTER MOST?")
print("-"*35)

importance = final_model.feature_importances_
feature_name = X.columns
featrue_importance = pd.DataFrame({
    'Feature': feature_name,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("\n Top 5 most important features: ")
print(featrue_importance.head().to_string(index=False))


plt.figure(figsize=(10, 6))
top_features = featrue_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel("Importance Score")
plt.title('Top 10 most important features for predicting house price')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches ='tight')

print('\n Feature importance chart saved as "feature_importance.png"')
plt.show()



