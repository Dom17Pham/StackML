import pandas as pd
import numpy as np
import os 
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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
stackoverflow_data = stackoverflow_data[stackoverflow_data['EdLevel'] != 'Something else']
stackoverflow_data = stackoverflow_data[stackoverflow_data['EdLevel'] != 'Some college/university study without earning a degree']

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

# Create a dictionary mapping education levels to ordinal values
education_levels = ['Primary/elementary school',
                    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
                    'Associate degree (A.A., A.S., etc.)',
                    'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
                    'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
                    'Professional degree (JD, MD, etc.)',
                    'Other doctoral degree (Ph.D., Ed.D., etc.)']

ordinal_values = [0, 1, 2, 3, 4, 5, 6]
education_mapping = {level: value for level, value in zip(education_levels, ordinal_values)}

# Apply ordinal encoding to the 'EdLevel' feature
x['EdLevel'] = x['EdLevel'].map(education_mapping)

# Function to handle non numerical data by converting each unique element into a int key 
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digits_vals = {}
        def convert_to_int(val):
            return text_digits_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digits_vals:
                    text_digits_vals[unique] = x
                    x+=1
            
            df[column] = list(map(convert_to_int, df[column]))

    return df

x = handle_non_numerical_data(x)

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [None, 50, 75, 125],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the RandomForestRegressor model
RF_model = RandomForestRegressor(random_state=1)

# Create the GridSearchCV instance
random_search = RandomizedSearchCV(RF_model, param_grid, n_iter=15, cv=5, scoring='neg_mean_absolute_error', n_jobs=4, random_state=1)

# Split data into training and validation data
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Perform random search
print('Randomized Search in Progress . . .')
start_time = time.time()
random_search.fit(train_x, train_y)
best_model = random_search.best_estimator_
search_elapsed_time = time.time() - start_time

# Make predictions
with tqdm(total=len(val_x), desc="Prediction Progress", unit="sample") as pbar_pred:
    RF_predictions = []
    start_time = time.time()
    for i in range(len(val_x)):
        prediction = best_model.predict(val_x.iloc[[i]])
        RF_predictions.append(prediction)
        pbar_pred.update(1)  # Update the prediction progress bar

    pred_elapsed_time = time.time() - start_time
    pbar_pred.set_postfix({"Elapsed Time": f"{pred_elapsed_time:.2f}s"})

Total_elapsed_time = search_elapsed_time + pred_elapsed_time
RF_predictions = np.concatenate(RF_predictions)
RF_mae = mean_absolute_error(val_y, RF_predictions)

# Printing the predictions
minutes = int(Total_elapsed_time // 60)
seconds = int(Total_elapsed_time % 60)
print(f"Total Elapsed Time for Predictions: {minutes} minutes {seconds} seconds")
print('Predictions:', RF_predictions)
print('MAE:', RF_mae, '\n')

# Plotting the results
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs. predicted values for Random Forest
plt.subplot(1, 2, 1)
plt.scatter(val_y, RF_predictions, alpha=0.5)
plt.plot(val_y, val_y, color='red', linestyle='--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Random Forest Model\nMAE: {:.2f}'.format(RF_mae))

plt.tight_layout()
plt.show()