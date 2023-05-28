import pandas as pd
import numpy as np
import os 
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from category_encoders import CatBoostEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tqdm import tqdm

# Loading the survey data into a dataframe
file_name = "StackOverflowSurvey2022.csv"
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

stackoverflow_data = pd.read_csv(filepath) 

# Drop rows where the compensation total (salary + bonuses + perks) is NA
stackoverflow_data = stackoverflow_data.dropna(subset=['CompTotal', 'Employment', 'EdLevel', 'YearsCode'])

# Drop rows where the education is not clearly defined
stackoverflow_data = stackoverflow_data[
    (stackoverflow_data['EdLevel'] != 'Something else') & 
    (stackoverflow_data['EdLevel'] != 'Some college/university study without earning a degree')                                     
]

# Drop rows where there is no employment, retirement, and not clearly defined
stackoverflow_data = stackoverflow_data[
    (~stackoverflow_data['Employment'].str.contains('Not employed')) &
    (stackoverflow_data['Employment'] != 'I prefer not to say') & 
    (stackoverflow_data['Employment'] != 'Retired')                                
]

# Prediction Target 
y = stackoverflow_data.CompTotal

# Calculate the IQR for salaries
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 0.15 * IQR
upper_bound = Q3 + 0.15 * IQR

# Filter out the outliers
outlier_mask = (y >= lower_bound) & (y <= upper_bound)
filtered_data = stackoverflow_data[outlier_mask]

# Specify the features to use for prediction model
features = ['Employment', 'EdLevel', 'YearsCode']

x = filtered_data[features]
y = filtered_data.CompTotal

# Replace all entries with 'Less than 1 year' to '0'
x = x.replace('Less than 1 year', '0')

# Replace all entries with 'More than 50 years' to '50'
x = x.replace('More than 50 years', '50')

# Convert the YearsCode dtype from object to int64
x.YearsCode = x['YearsCode'].astype('int64')

# Perform CatBoost encoding for the categorical features
encoder = CatBoostEncoder(cols=['Employment', 'EdLevel'])
x = encoder.fit_transform(x, y)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Create the XGBRegressor model
XGB_model = xgb.XGBRegressor(n_jobs=4,random_state=1)

# Create the GridSearchCV instance
random_search = RandomizedSearchCV(XGB_model, param_grid, n_iter=15, cv=5, scoring='neg_mean_absolute_error', n_jobs=4, random_state=1)

# Split data into training and validation data
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Perform random search
print('Randomized Search in Progress . . .')
start_time = time.time()
random_search.fit(train_x, train_y)
best_model = random_search.best_estimator_
search_elapsed_time = time.time() - start_time

# Get feature importance scores
importance_scores = best_model.feature_importances_
feature_names = train_x.columns

# Make predictions
with tqdm(total=len(val_x), desc="Prediction Progress", unit="sample") as pbar_pred:
    XGB_predictions = []
    start_time = time.time()
    for i in range(len(val_x)):
        prediction = best_model.predict(val_x.iloc[[i]])
        XGB_predictions.append(prediction)
        pbar_pred.update(1)  # Update the prediction progress bar

    pred_elapsed_time = time.time() - start_time
    pbar_pred.set_postfix({"Elapsed Time": f"{pred_elapsed_time:.2f}s"})

Total_elapsed_time = search_elapsed_time + pred_elapsed_time
XGB_predictions = np.array(XGB_predictions)
RF_mae = mean_absolute_error(val_y, XGB_predictions)

# Sort feature importance scores in descending order
sorted_indices = importance_scores.argsort()[::-1]
sorted_scores = importance_scores[sorted_indices]
sorted_names = feature_names[sorted_indices]

# Printing the predictions
minutes = int(Total_elapsed_time // 60)
seconds = int(Total_elapsed_time % 60)
print(f"Total Elapsed Time for Predictions: {minutes} minutes {seconds} seconds")
print('Predictions:', XGB_predictions)
print('MAE:', RF_mae, '\n')

# Plotting the results
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs. predicted values 
plt.subplot(1, 2, 1)
plt.scatter(val_y, XGB_predictions, alpha=0.5)
plt.plot(val_y, val_y, color='red', linestyle='--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('XGBoost Model\nMAE: {:.2f}'.format(RF_mae))

# Plot feature importance
plt.subplot(1, 2, 2)
plt.bar(range(len(sorted_scores)), sorted_scores)
plt.xticks(range(len(sorted_scores)), sorted_names, rotation='vertical')
plt.xlabel('Features')
plt.ylabel('Importance Scores')
plt.title('Feature Importance')

plt.tight_layout()
plt.show()