import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Loading the data
file_name = "StackOverflowSurvey2022.csv"
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

stackoverflow_data = pd.read_csv(filepath)

# Drop any row where the compensation total (salary + bonuses + perks) is NA
# This will remove missing values that might affect data accuracy 
stackoverflow_data = stackoverflow_data.dropna(subset=['CompTotal', 'Employment', 'EdLevel', 'YearsCode'])

# Prediction Target
y = stackoverflow_data.CompTotal

# Calculate the IQR for salaries
Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

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

# Split data into training and validation data
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Random Forest 
RF_model = RandomForestRegressor(random_state=1)
RF_model.fit(train_x, train_y)
RF_predictions = RF_model.predict(val_x)

# Linear Regression
LR_model = LinearRegression()
LR_model.fit(train_x, train_y)
LR_predictions = LR_model.predict(val_x)

# Results 
print('Random Forest model:')
print('Predictions:', RF_predictions)
print('MAE:', mean_absolute_error(val_y, RF_predictions), '\n')

print('Linear Regression model:')
print('Predictions:', LR_predictions)
print('MAE:', mean_absolute_error(val_y, LR_predictions), '\n')