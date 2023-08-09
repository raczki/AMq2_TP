"""
hyperparameters.py

DESCRIPTION: This script defines a pipeline for hyperparameter tuning using the Hyperopt library. It loads a dataset, preprocesses it,
trains an XGBoost regression model, and performs hyperparameter tuning to find the best set of hyperparameters.
AUTHOR: Karen Raczkowski
DATE: 18/8/23
"""

# Imports
import logging
import os
import xgboost as xgboost_regressor
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp
from hyperopt.pyll import scope

def pre_processing(pandas_df: pd.DataFrame):
    """
    Preprocesses the given DataFrame by dropping unnecessary columns and splitting into train and test sets.

    Args:
        pandas_df (pd.DataFrame): The DataFrame containing raw data.

    Returns:
        pd.DataFrame, pd.DataFrame: Processed train and test DataFrames.
    """
    dataset = pandas_df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

    # Split the dataset into train and test sets
    df_train = dataset.loc[pandas_df['Set'] == 'train']
    df_test = dataset.loc[pandas_df['Set'] == 'test']

    return df_train, df_test

class HyperParametersPipeline(object):
    """
    A pipeline for hyperparameter tuning using the Hyperopt library.

    Attributes:
        input_path (str): The path to the input data file.
        output_path (str): The path where outputs will be saved.
    """
    def __init__(self, input_path, output_path: str = None):
        """
        Initialize the HyperParametersPipeline object.

        Args:
            input_path (str): The path to the input data file.
            output_path (str, optional): The path to save outputs. Defaults to None.
        """
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the specified input path.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            train_file = 'dataframe.csv'
            train_data = os.path.join(self.input_path, train_file)
            pandas_df = pd.read_csv(train_data)
            logging.info("Loading data from:  {self.input_path}")

        except (FileNotFoundError, PermissionError, OSError) as error_load_file:
            logging.exception(
                "An error occurred while loading data: %s", error_load_file)

        return pandas_df

    def model_to_train(self, df: pd.DataFrame):
        """
        Prepare the dataset for model training.

        Args:
            df (pd.DataFrame): The DataFrame containing the dataset.

        Returns:
            np.array, np.array: Arrays of features and target variable for training.
        """

        global x_train
        global y_train

        df_train, df_test = pre_processing(pandas_df=df)

        # Deleting columns without data
        df_train.drop(['Unnamed: 0', 'Set'], axis=1, inplace=True)
        df_test.drop(['Unnamed: 0', 'Item_Outlet_Sales', 'Set'],
                     axis=1, inplace=True)

        seed = 28

        # Splitting of  the dataset in training and validation sets
        X = df_train.drop(columns='Item_Outlet_Sales')
        y = df_train['Item_Outlet_Sales']
        x_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=seed)
        return x_train, y_train

    def train_model(self, x_train, y_train):
        """
        Train the XGBoost model and perform cross-validation.

        Args:
            x_train (np.array): Array of input features for training.
            y_train (np.array): Array of target values for training.

        Returns:
            np.array: Array of cross-validation scores.
        """
        seed = 28
        model_trained = xgboost_regressor.XGBRegressor(
            objective='reg:linear', n_estimators=10, random_state=seed)
        # Train the model
        score_model = cross_val_score(
            model_trained, x_train, y_train, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10)
        print('Score model', score_model)
        print(np.mean(score_model), np.std(score_model))
        return score_model

    def return_score(self, params):
        """
        Return the negative root mean squared error score using given hyperparameters.

        Args:
            params (dict): Dictionary of hyperparameters.

        Returns:
            float: Negative root mean squared error score.
        """
        model = xgboost_regressor.XGBRegressor(**params)
        root_mean_square_error = -np.mean(cross_val_score(
            model, x_train, y_train, cv=4, n_jobs=-1, scoring='neg_root_mean_squared_error'))
        return root_mean_square_error

    def objective(self, params):
        """
        Objective function for hyperparameter tuning using negative root mean squared error.

        Args:
            params (dict): Dictionary of hyperparameters.

        Returns:
            float: Negative root mean squared error score.
        """
        return self.return_score(params)

    def run(self):
        """
        Run the hyperparameter tuning pipeline.

        Returns:
            dict: Dictionary of best hyperparameters found.
        """
        data_frame = self.load_data()
        x_trained, y_trained = self.model_to_train(data_frame)
        
        # Define the initial search space with broader ranges
        space = {
            'lambda': hp.uniform('lambda', 0.1, 20.0), # L2 regularization term (reg_lambda)
            'alpha': hp.uniform('alpha', 0.1, 20.0), # L1 regularization term (reg_alpha)
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.5), 
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1.0), # Feature subsampling ratio
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 1)) # Number of boosting rounds
        }

        # Max_evals should be adjusted based on the exploration strategy
        best_hyperparams = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=200) 
        print("Best hyperparameters:", best_hyperparams)
        return best_hyperparams

if __name__ == "__main__":
    pipeline = HyperParametersPipeline(
        input_path='../data/',
        output_path='../hp')

    # Retrieve the best hyperparameters found by Hyperopt.
    best_hyperparams = pipeline.run()
    print("Best hyperparameters:", best_hyperparams)
