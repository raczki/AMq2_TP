"""
train.py

This script contains functions and classes for training a machine learning model 
and evaluating its performance.

DESCRIPTION: This script includes functions to save DataFrames as CSV files, 
evaluate model performance using various metrics and visualizations,
and implement a model training pipeline.
AUTHOR: Karen Raczkowski
DATE: 18/8/23
"""

# Imports
import os
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import probplot


def save_csv(dataframe_train: pd.DataFrame, dataframe_test: pd.DataFrame):
    """
    Save DataFrame train and test to CSV files.

    Args:
        dataframe_train (pd.DataFrame): Dataframe for training.
        dataframe_test (pd.DataFrame): Datafrane for testing.

    Returns:
        None
    """
    try:
        out_path = '../model'
        train_file = 'train_final.csv'
        test_file = 'test_final.csv'
        output_train = os.path.join(out_path, train_file)
        output_test = os.path.join(out_path, test_file)
        dataframe_train.to_csv(output_train)
        logging.info("Writing train dataset: %s", output_train)
        print('Writing train dataset...')
        dataframe_test.to_csv(output_test)
        logging.info("Writing test dataset: %s", output_test)
        print('Writing test dataset...')
    except (IOError, OSError) as error:
        logging.error("Error writing to file: %s", error)
        print(f"Error writing to file: {error}")
    except Exception as error:
        logging.error("An unexpected error occurred: %s", error)
        print(f"An unexpected error occurred: {error}")

    return train_file, test_file


def evaluate_model_performance(train_x, train_y, y_value, x_value, pred, model, model_path):
    """
    Evaluate the performance of a machine learning model using various metrics and visualizations.

    Args:
        train_x (pd.DataFrame): Features of the training set.
        train_y (pd.Series): Target values of the training set.
        y_value (pd.Series): True target values for validation.
        x_value (pd.DataFrame): Features of the validation set.
        pred (pd.Series): Predicted target values for validation.
        model (object): The trained machine learning model.

    Returns:
        None
    """
    mse_train = metrics.mean_squared_error(train_y, model.predict(train_x))
    mse_training = mse_train**2
    r2_train = model.score(train_x, train_y)
    print('Model evaluation metrics:')
    print(f'Training RMSE: {mse_training:.2f} - R2: {r2_train:.4f}')

    mse_val = metrics.mean_squared_error(y_value, pred)
    mse_validation = mse_val**2
    r2_val = model.score(x_value, y_value)
    model_intercept = model.intercept_
    print(f'Validation RMSE: {mse_validation:.2f} - R2: {r2_val:.4f}')

    print('\nAdditional metrics:')

    mae_train = metrics.mean_absolute_error(train_y, model.predict(train_x))
    mae_val = metrics.mean_absolute_error(y_value, pred)
    print(f'Training MAE: {mae_train:.2f}')
    print(f'Validation MAE: {mae_val:.2f}')

    mape_train = metrics.mean_absolute_percentage_error(
        train_y, model.predict(train_x))
    mape_val = metrics.mean_absolute_percentage_error(y_value, pred)
    print(f'Training MAPE: {mape_train:.2f}%')
    print(f'Validation MAPE: {mape_val:.2f}%')

    print('\nModel coefficients:')

    # Model intercept
    print(f'Intercept: {model_intercept:.2f}')

    coef = pd.DataFrame(train_x.columns, columns=['features'])
    coef['Estimated coefficients'] = model.coef_
    print(coef, '\n')

    print('Saving visualizations...')

    bar_plot = coef.sort_values(by='Estimated coefficients').set_index('features').plot(
        kind='bar', title='Importance of variables', figsize=(12, 6))
    bar_plot.figure.savefig(os.path.join(model_path, 'bar_plot.png'))
    plt.close()

    # Residuals Plot
    residuals = y_value - pred
    residuals_plot = plt.figure(figsize=(10, 6))
    sns.residplot(x=pred, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    residuals_plot.savefig(os.path.join(model_path, 'residuals_plot.png'))
    plt.close()

    # QQ Plot
    qq_plot = plt.figure(figsize=(10, 6))
    probplot(residuals, plot=plt)
    plt.title('QQ Plot of Residuals')
    qq_plot.savefig(os.path.join(model_path, 'qq_plot.png'))
    plt.close()


class ModelTrainingPipeline:
    """
    A class for training a machine learning model using a provided dataset,
    and saving the trained model to a file.

    Attributes:
    input_path (str): The path to the directory containing the input dataset.
    model_path (str): The path to the directory where the trained model will be saved.
    logger (logging.Logger): The logger instance for logging messages.

    Methods:
    read_data():
        Read data from the specified input path and return it as a DataFrame.
    model_training(data_frame):
        Train a machine learning model using the provided DataFrame.
    model_dump(model_trained):
        Save a trained machine learning model to a pickle file.
    run():
        Run the pipeline to read data, train a model, and save the trained model.
    """

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)

    def read_data(self) -> pd.DataFrame:
        """
        Read data from the specified input path and return it as a DataFrame.

        This function reads data from the provided input path, 
        assuming it's in CSV format, and returns a pandas DataFrame.

        :return: The DataFrame containing the loaded data.
        :rtype: pd.DataFrame
        """

        try:
            train_file = 'dataframe.csv'
            train_data = os.path.join(self.input_path, train_file)
            logging.info("Loading data from: %s", self.input_path)
            pandas_df = pd.read_csv(train_data)

        except FileNotFoundError:
            logging.error("File not found: %s", train_data)
        except PermissionError:
            logging.error("Permission error while accessing: %s", train_data)
        except OSError:
            logging.error("OS error while accessing: %s", train_data)
        except pd.errors.ParserError:
            logging.error("Error parsing the CSV file: %s", train_data)
        except Exception as error:
            logging.error(
                "An error occurred while loading data: %s", str(error))

        return pandas_df

    def model_training(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Train a machine learning model using the provided DataFrame.

        This function takes a DataFrame 'df' containing input features and target values. 
        It preprocesses the data, splits it into training and validation sets, 
        trains a LinearRegression model, and evaluates its performance using various metrics.

        :param df: The DataFrame containing input features and target values.
        :type df: pd.DataFrame
        :return: The trained LinearRegression model.
        :rtype: LinearRegression
        """

        # df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels=[1, 2, 3, 4])
        print('Item_MRP', data_frame['Item_MRP'])
        dataset = data_frame.drop(
            columns=['Item_Identifier', 'Outlet_Identifier'])
        print(dataset.info())

        # Split of the dataset in train y test sets
        df_train = dataset.loc[data_frame['Set'] == 'train']
        df_test = dataset.loc[data_frame['Set'] == 'test']

        # Deleting columns without data
        df_train.drop(['Unnamed: 0', 'Set'], axis=1, inplace=True)
        df_test.drop(['Unnamed: 0', 'Item_Outlet_Sales', 'Set'],
                     axis=1, inplace=True)

        # Writing the model in a file
        save_csv(df_train, df_test)

        seed = 28
        model = LinearRegression()

        # Splitting of  the dataset in training and validation sets
        features = df_train.drop(columns='Item_Outlet_Sales')
        print('Este es el valor de X')
        features.info()
        target = df_train['Item_Outlet_Sales']
        x_train, x_val, y_train, y_val = train_test_split(
            features, target, test_size=0.3, random_state=seed)

        # Training the model
        trained_model = model.fit(x_train, y_train)
        print(x_val)
        predicted_model = model.predict(x_val)

        evaluate_model_performance(
            x_train, y_train, y_val, x_val, predicted_model, model, self.model_path)

        return trained_model

    def model_dump(self, model_trained) -> None:
        """
        Save a trained machine learning model to a pickle file.

        Args:
            model_trained (object): The trained machine learning model.

        Returns:
            None
        """

        try:
            trained_file = "trained_model.pkl"
            output_path = os.path.join(self.model_path, trained_file)
            print('Writing pickle file....')

            with open(output_path, 'wb') as model_output:
                pickle.dump(model_trained, model_output)

            self.logger.info("Pickle file saved successfully.")

        except FileNotFoundError:
            print(
                f"Error: The specified directory '{self.model_path}' does not exist.")
        except IOError as error:
            print("Error writing to file:", error)
        except Exception as error:
            print("An unexpected error occurred:", error)

    def run(self):
        """
        Run the pipeline to read data, train a model, and save the trained model.

        This function orchestrates the different steps of the pipeline:
        1. Reads data using the read_data() method.
        2. Trains a model using the model_training() method.
        3. Saves the trained model using the model_dump() method.
        """
        data_frame = self.read_data()
        model_trained = self.model_training(data_frame)
        self.model_dump(model_trained)


if __name__ == "__main__":

    ModelTrainingPipeline(input_path='../data/',
                          model_path='../model').run()
