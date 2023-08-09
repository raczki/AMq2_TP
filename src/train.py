"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
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
        logging.info(f"Writing dataframe train: {output_train}")
        print('Writing dataframe train...')
        dataframe_test.to_csv(output_test)
        logging.info(f"Writing dataframe test: {output_test}")
        print('Writing dataframe test...')
    except IOError as e:
        logging.error(f"Error writing to file: {e}")
        print(f"Error writing to file: {e}")
    except OSError as e:
        logging.error(f"Error in file system operation: {e}")
        print(f"Error in file system operation: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

    return train_file, test_file

def evaluate_model_performance(train_x, train_y, y_value, x_value, pred, model):
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
    print(
        f'TRAINING: RMSE: {mse_training:.2f} - R2: {r2_train:.4f}')

    mse_val = metrics.mean_squared_error(y_value, pred)
    mse_validation = mse_val**2
    r2_val = model.score(x_value, y_value)
    model_intercept = model.intercept_
    print(f'VALIDATION: RMSE: {mse_validation:.2f} - R2: {r2_val:.4f}')

    print('\nAdditional metrics:')
    
    mae_train = metrics.mean_absolute_error(train_y, model.predict(train_x))
    mae_val = metrics.mean_absolute_error(y_value, pred)
    print(f'TRAINING: MAE: {mae_train:.2f}')
    print(f'VALIDATION: MAE: {mae_val:.2f}')
    
    mape_train = metrics.mean_absolute_percentage_error(train_y, model.predict(train_x))
    mape_val = metrics.mean_absolute_percentage_error(y_value, pred)
    print(f'TRAINING: MAPE: {mape_train:.2f}%')
    print(f'VALIDATION: MAPE: {mape_val:.2f}%')

    print('\nModel coefficients:')
    
    # Model intercept
    print(f'Intercept: {model_intercept:.2f}')

    coef = pd.DataFrame(train_x.columns, columns=['features'])
    coef['Estimated coefficients'] = model.coef_
    print(coef, '\n')
    coef.sort_values(by='Estimated coefficients').set_index('features').plot(
        kind='bar', title='Importance of variables', figsize=(12, 6))
    plt.show()

    print('\nVisualizations:')

    # Residuals Plot
    residuals = y_value - pred
    plt.figure(figsize=(10, 6))
    sns.residplot(x=pred, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()
    
    # QQ Plot
    plt.figure(figsize=(10, 6))
    probplot(residuals, plot=plt)
    plt.title('QQ Plot of Residuals')
    plt.show()

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        Read data from the specified input path and return it as a DataFrame.

        This function reads data from the provided input path, assuming it's in CSV format, and returns a pandas DataFrame.

        :return: The DataFrame containing the loaded data.
        :rtype: pd.DataFrame
        """
            
        try:
            train_file = 'dataframe.csv'
            train_data = os.path.join(self.input_path, train_file)
            logging.info("Loading data from: {}".format(self.input_path))
            pandas_df = pd.read_csv(train_data)

        except FileNotFoundError:
            logging.error("File not found: {}".format(train_data))
        except PermissionError:
            logging.error("Permission error while accessing: {}".format(train_data))
        except OSError:
            logging.error("OS error while accessing: {}".format(train_data))
        except pd.errors.ParserError:
            logging.error("Error parsing the CSV file: {}".format(train_data))
        except Exception as e:
            logging.error("An error occurred while loading data: {}".format(str(e)))

        return pandas_df

    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train a machine learning model using the provided DataFrame.

        This function takes a DataFrame 'df' containing input features and target values. It preprocesses the data, splits it
        into training and validation sets, trains a LinearRegression model, and evaluates its performance using various metrics.

        :param df: The DataFrame containing input features and target values.
        :type df: pd.DataFrame
        :return: The trained LinearRegression model.
        :rtype: LinearRegression
        """
        
        # df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels=[1, 2, 3, 4])
        print('Item_MRP', df['Item_MRP'])
        dataset = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
        print(dataset.info())

        # Split of the dataset in train y test sets
        df_train = dataset.loc[df['Set'] == 'train']
        df_test = dataset.loc[df['Set'] == 'test']

        # Deleting columns without data
        df_train.drop(['Unnamed: 0', 'Set'], axis=1, inplace=True)
        df_test.drop(['Unnamed: 0', 'Item_Outlet_Sales', 'Set'],
                     axis=1, inplace=True)

        # Writing the model in a file
        save_csv(df_train, df_test)

        seed = 28
        model = LinearRegression()

        # Splitting of  the dataset in training and validation sets
        X = df_train.drop(columns='Item_Outlet_Sales')
        print('Este es el valor de X')
        X.info()
        y = df_train['Item_Outlet_Sales']
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=seed)

        # Training the model
        trained_model = model.fit(x_train, y_train)
        print(x_val)
        predicted_model = model.predict(x_val)

        evaluate_model_performance(x_train, y_train, y_val, x_val, predicted_model, model)

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
            model_output = open(os.path.join(self.model_path, trained_file), 'wb')
            print('Writing pickle file....')
            pickle.dump(model_trained, model_output)
            model_output.close()
            self.logger.info(f'Pickle file saved successfully.')

        except FileNotFoundError as e:
            print(f"Error: The specified directory '{self.model_path}' does not exist.")
        except IOError as e:
            print(f"Error writing to file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return None

    def run(self):

        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = './data/',
                          model_path = './model').run()