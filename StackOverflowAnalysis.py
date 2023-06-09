import pandas as pd
import numpy as np
import os 
import time
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from category_encoders import CatBoostEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tqdm import tqdm

# Load survey datasets into a dataframe 
directory = os.path.dirname(os.path.abspath(__file__))
stackoverflow_data = pd.DataFrame()

for file_name in os.listdir(directory):
    if file_name.endswith(".csv"):
        filepath = os.path.join(directory, file_name)
        df = pd.read_csv(filepath)
        stackoverflow_data = pd.concat([stackoverflow_data, df], ignore_index=True)

# Drop rows where the specified columns are NA in any of these columns
stackoverflow_data = stackoverflow_data.dropna(subset=['ConvertedCompYearly', 'Employment', 'EdLevel', 'YearsCode', 'DevType'])

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
y = stackoverflow_data.ConvertedCompYearly

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
features = ['Employment', 'EdLevel', 'YearsCode', 'DevType']

x = filtered_data[features]
y = filtered_data.ConvertedCompYearly

# Replace all entries with 'Less than 1 year' to '0'
x = x.replace('Less than 1 year', '0')

# Replace all entries with 'More than 50 years' to '50'
x = x.replace('More than 50 years', '50')

# Convert the YearsCode dtype from object to int64
x.YearsCode = x['YearsCode'].astype('int64')

# Perform CatBoost encoding for the categorical features
encoder = CatBoostEncoder(cols=['Employment', 'EdLevel', 'DevType'])
x = encoder.fit_transform(x, y)

def objective(trial):
        # Define the hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    subsample = trial.suggest_float('subsample', 0.8, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.8, 1.0)

    # Build and train your XGBoost model using the hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    # Train and evaluate the model
    model.fit(train_x, train_y)
    y_pred = model.predict(val_x)
    score = mean_absolute_error(val_y, y_pred)

    return score

# Split data into training and validation data
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Set up the Optuna study
start_time = time.time()
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
trial_elapsed_time = time.time() - start_time

# Get the best hyperparameters
best_params = study.best_params

# Build the XGBoost model using the best hyperparameters
best_model = xgb.XGBRegressor(**best_params)

# Train the best model on the entire training data
best_model.fit(train_x, train_y)

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

XGB_predictions = np.array(XGB_predictions)
RF_mae = mean_absolute_error(val_y, XGB_predictions)
Total_elapsed_time = trial_elapsed_time + pred_elapsed_time

# Sort feature importance scores in descending order
sorted_indices = importance_scores.argsort()[::-1]
sorted_scores = importance_scores[sorted_indices]
sorted_names = feature_names[sorted_indices]

# Printing the predictions
print('\nPredictions:\n', XGB_predictions, '\n')

# Print performance metrics
print("--- PERFORMANCE REPORT --- ")
print('MAE:', RF_mae)
print(f"Elapsed Time for Optuna study: {int(trial_elapsed_time // 60)} minutes {int(trial_elapsed_time % 60)} seconds")
print(f"Elapsed Time for Predictions: {int(pred_elapsed_time // 60)} minutes {int(pred_elapsed_time % 60)} seconds")
print(f"Total Elapsed Time: {int(Total_elapsed_time // 60)} minutes {int(Total_elapsed_time % 60)} seconds")

# Plotting the results
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs. predicted values 
plt.subplot(1, 2, 1)
plt.scatter(val_y, XGB_predictions, alpha=0.5)
plt.plot(val_y, val_y, color='red', linestyle='--')
plt.xlabel('Actual Yearly Salary (USD)')
plt.ylabel('Predicted Yearly Salary (USD)')
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