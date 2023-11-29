import pandas as pd
import numpy as np
import os 
import time
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from category_encoders import CatBoostEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

default_num_of_thread = 1
num_threads = default_num_of_thread
# Create window 
window=Tk()
window.configure(bg='grey')
window.title("Software Industry Salary Predictor")
window.geometry("1024x768")

# Load survey datasets into a dataframe 
directory = os.path.dirname(os.path.abspath(__file__))
stackoverflow_data = pd.DataFrame()

for file_name in os.listdir(directory):
    if file_name.endswith(".csv"):
        filepath = os.path.join(directory, file_name)
        df = pd.read_csv(filepath)
        stackoverflow_data = pd.concat([stackoverflow_data, df], ignore_index=True)

# Split data into training and testing data
stack_train, stack_test = train_test_split(stackoverflow_data, test_size=0.3)

#################################################
# TRAINING DATA PREPROCESSING                   #
#################################################

# Drop rows where the specified columns are NA in any of these columns
stack_train = stack_train.dropna(subset=['ConvertedCompYearly','Employment', 'EdLevel', 'YearsCode', 'DevType', 'Country'])

# Only keep entries where the country is U.S or Canada
stack_train = stack_train[
    (stack_train['Country'] == 'Canada') |
    (stack_train['Country'] == 'United States of America')                           
]

# Drop rows where the education is not clearly defined
stack_train = stack_train[
    (stack_train['EdLevel'] != 'Something else') & 
    (stack_train['EdLevel'] != 'Some college/university study without earning a degree')                                     
]

# Only keep entries from full-time employees
stack_train = stack_train[
    (stack_train['Employment'] == 'Employed, full-time')                                
]

# Only keep entries with the following dev type
stack_train = stack_train[
    (stack_train['DevType'] == 'Developer, full-stack') |     
    (stack_train['DevType'] == 'Developer, back-end') | 
    (stack_train['DevType'] == 'Developer, front-end') | 
    (stack_train['DevType'] == 'Developer, mobile') | 
    (stack_train['DevType'] == 'Developer, QA or test') | 
    (stack_train['DevType'] == 'Developer, game or graphics') | 
    (stack_train['DevType'] == 'Developer, desktop or enterprise applications') 
]

# Prediction Target 
stack_train_y = stack_train.ConvertedCompYearly

# Extracting the features to use for prediction model
features = ['EdLevel', 'YearsCode', 'DevType']

# Calculate the IQR for salaries
Q1 = np.percentile(stack_train_y, 25)
Q3 = np.percentile(stack_train_y, 75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 0.15 * IQR
upper_bound = Q3 + 0.15 * IQR

# Filter out the outliers
outlier_mask = (stack_train_y >= lower_bound) & (stack_train_y <= upper_bound)
filtered_data = stack_train[outlier_mask]

stack_train_x = filtered_data[features]
stack_train_y = filtered_data.ConvertedCompYearly

# Replace all entries with 'Less than 1 year' to '0'
stack_train_x = stack_train_x.replace('Less than 1 year', '0')

# Replace all entries with 'More than 50 years' to '50'
stack_train_x = stack_train_x.replace('More than 50 years', '50')

# Convert the YearsCode dtype from object to int64
stack_train_x.YearsCode = stack_train_x['YearsCode'].astype('int64')

# Perform CatBoost encoding for the categorical features
encoder = CatBoostEncoder(cols=['EdLevel', 'DevType'])
stack_train_x = encoder.fit_transform(stack_train_x, stack_train_y)

###############################
# TESTING DATA PREPROCESSING #
###############################

# Drop rows where the specified columns are NA in any of these columns
stack_test = stack_test.dropna(subset=['ConvertedCompYearly','Employment', 'EdLevel', 'YearsCode', 'DevType','Country'])

# Only keep entries where the country is U.S or Canada
stack_test = stack_test[
    (stack_test['Country'] == 'Canada') |
    (stack_test['Country'] == 'United States of America')                           
]

# Drop rows where the education is not clearly defined
stack_test = stack_test[
    (stack_test['EdLevel'] != 'Something else') & 
    (stack_test['EdLevel'] != 'Some college/university study without earning a degree')                                     
]

# Only keep entries from full-time employees
stack_test = stack_test[
    (stack_test['Employment'] == 'Employed, full-time')                             
]

# Only keep entries with the following dev type
stack_test = stack_test[
    (stack_test['DevType'] == 'Developer, full-stack') |     
    (stack_test['DevType'] == 'Developer, back-end') | 
    (stack_test['DevType'] == 'Developer, front-end') | 
    (stack_test['DevType'] == 'Developer, mobile') | 
    (stack_test['DevType'] == 'Developer, QA or test') | 
    (stack_test['DevType'] == 'Developer, game or graphics') | 
    (stack_test['DevType'] == 'Developer, desktop or enterprise applications') 
]

# Prediction Target 
stack_test_y = stack_test.ConvertedCompYearly

# Extracting the features to use for prediction model
features = ['EdLevel', 'YearsCode', 'DevType']

# Calculate the IQR for salaries
Q1 = np.percentile(stack_test_y, 25)
Q3 = np.percentile(stack_test_y, 75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 0.15 * IQR
upper_bound = Q3 + 0.15 * IQR

# Filter out the outliers
outlier_mask = (stack_test_y >= lower_bound) & (stack_test_y <= upper_bound)
filtered_data = stack_test[outlier_mask]

stack_test_x = filtered_data[features]
stack_test_y = filtered_data.ConvertedCompYearly

# Replace all entries with 'Less than 1 year' to '0'
stack_test_x = stack_test_x.replace('Less than 1 year', '0')

# Replace all entries with 'More than 50 years' to '50'
stack_test_x = stack_test_x.replace('More than 50 years', '50')

# Convert the YearsCode dtype from object to int64
stack_test_x.YearsCode = stack_test_x['YearsCode'].astype('int64')

# Perform CatBoost encoding for the categorical features
encoder = CatBoostEncoder(cols=['EdLevel', 'DevType'])
stack_test_x = encoder.fit_transform(stack_test_x, stack_test_y)

##################
# MODEL TRAINING #
##################

def objective(trial):
    # Define the hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 5, 15)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
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
    model.fit(stack_train_x, stack_train_y)
    y_pred = model.predict(stack_test_x)
    score = mean_absolute_error(stack_test_y, y_pred)

    return score

def start_study():
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    start_time = time.time()
    num_trials = int(num_trials_entry.get()) 
    num_threads = int(num_threads_entry.get()) 
    study.optimize(objective, n_trials=num_trials, n_jobs=num_threads)
    trial_elapsed_time = time.time() - start_time
    best_params = study.best_params

    # Build the XGBoost model using the best hyperparameters
    best_model = xgb.XGBRegressor(**best_params)

    # Train the best model on the entire training data
    best_model.fit(stack_train_x, stack_train_y)

    # Make predictions
    XGB_predictions = best_model.predict(stack_test_x)

    # MAE calculations
    MAE = mean_absolute_error(stack_test_y, XGB_predictions)

    # Sort feature importance scores in descending order
    importance_scores = best_model.feature_importances_
    sorted_indices = importance_scores.argsort()[::-1]
    sorted_scores = importance_scores[sorted_indices]
    sorted_names = stack_train_x.columns[sorted_indices]

    # Display performance metrics
    performance_label.config(text='--- PERFORMANCE REPORT ---\nMAE: {:.2f}\nElapsed Time for Optuna study: {} minutes {} seconds'.format(
        MAE, int(trial_elapsed_time // 60), int(trial_elapsed_time % 60)))

    # Display the hyperparameters used
    hyperparameters_label.config(text='--- BEST HYPERPARAMETERS ---\n' + '\n'.join(f"{key}: {value}" for key, value in best_params.items()))

    # Plot the results
    fig = plt.figure(figsize=(10, 6))

    # Scatter plot of actual vs. predicted values
    ax1 = fig.add_subplot(121)
    ax1.scatter(stack_test_y, XGB_predictions, alpha=0.5)
    ax1.plot(stack_test_y, stack_test_y, color='red', linestyle='--')
    ax1.set_xlabel('Actual Yearly Salary (USD)')
    ax1.set_ylabel('Predicted Yearly Salary (USD)')
    ax1.set_title('XGBoost Model\nMAE: {:.2f}'.format(MAE))

    # Plot feature importance
    ax2 = fig.add_subplot(122)
    ax2.bar(range(len(sorted_scores)), sorted_scores)
    ax2.set_xticks(range(len(sorted_scores)))
    ax2.set_xticklabels(sorted_names, rotation='vertical')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Importance Scores')
    ax2.set_title('Feature Importance')

    # Create a canvas to display the plot in Tkinter
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Restart the study and clear the existing results
def restart_study():
    performance_label.config(text='--- PERFORMANCE REPORT ---')
    hyperparameters_label.config(text='--- BEST HYPERPARAMETERS ---')
    start_study()

# Create GUI elements
start_button = Button(window, text='Start Study', command=start_study)
start_button.pack()

restart_button = Button(window, text='Restart Study', command=restart_study)
restart_button.pack()

# Number of trials input field
num_trials_frame = Frame(window)
num_trials_frame.pack()
num_trials_label = Label(num_trials_frame, text='Enter number of trials:')
num_trials_label.pack(side=LEFT)
num_trials_entry = Entry(num_trials_frame)
num_trials_entry.pack(side=LEFT)

# Number of threads input field
num_threads_frame = Frame(window)
num_threads_frame.pack()
num_threads_label = Label(num_threads_frame, text='Enter number of CPU Threads:')
num_threads_label.pack(side=LEFT)
num_threads_entry = Entry(num_threads_frame)
num_threads_entry.pack(side=LEFT)

performance_label = Label(window, text='--- PERFORMANCE REPORT ---')
performance_label.pack()

hyperparameters_label = Label(window, text='--- BEST HYPERPARAMETERS ---')
hyperparameters_label.pack()

window.protocol("WM_DELETE_WINDOW", window.quit)

window.mainloop()