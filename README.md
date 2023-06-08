# StackML: Machine Learning Project 

## Project Description 
This project utilizes the Stack Overflow 2022 Developer Survey data to build a machine learning model that employs the XGBoost algorithm 
to predict salaries based on various features such as employment status, education level, years of coding experience, and developer roles. 

The goal is to provide insights into the factors that influence salary levels in the software development industry.

## Getting Started
1) Create and activate a virtual environment:

>Windows
``` 
python -m venv <your-env-name>       # Create a virtual environment
<your-env-name>\Scripts\activate     # Activate the virtual environment 
```
>Linux/Mac
```
python -m venv <your-env-name>       # Create a virtual environment
source <your-env-name>/bin/activate  # Activate the virtual environment 
```

2) Download the following files from the repository into your venv directory:
* StackOverflowAnalysis.py
* requirements.txt


3) [Download survey dataset](https://insights.stackoverflow.com/survey/) into your venv directory

4) Install the required dependencies into your environment:
```
pip install -r requirements.txt
```
5) Run the StackOverflowAnalysis.py script:
```
python StackOverflowAnalysis.py
```
## Pipeline 
These steps are implemented in the StackOverflowAnalysis.py script.

### Data Preprocessing 
The data preprocessing steps include handling missing values, filtering outliers, encoding categorical features, and scaling numerical features. 

### Model Training 
The machine learning model used for this project is an XGBoost regressor. 
The model is trained using the preprocessed data and hyperparameters are tuned using randomized search cross-validation. 

### Evaluation 
The model's performance is evaluated using mean absolute error (MAE) as the evaluation metric. 
The predictions are compared with the actual salary values from the validation dataset. 
The evaluation results and analysis are presented through plotted graphs.
