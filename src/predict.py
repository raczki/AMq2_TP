"""
predict.py

This script defines a pipeline for making predictions using a trained machine learning model. It loads input data, loads a
trained model, makes predictions on the input data, and writes the predictions to a CSV file.

AUTHOR: Karen Raczkowski
DATE: 18/8/23
"""

# Imports
import pandas as pd
import joblib
import logging

class MakePredictionPipeline(object):
    """
    A pipeline for making predictions using a trained machine learning model.

    Attributes:
        input_path (str): The path to the input data file.
        output_path (str): The path where the predicted data will be saved.
        model_path (str): The path to the trained model file.
    """

    def __init__(self, input_path, output_path, model_path: str = None):
        """
        Initialize the MakePredictionPipeline object.

        :param input_path: The path to the input data file.
        :type input_path: str
        :param output_path: The path to save the predicted data.
        :type output_path: str
        :param model_path: The path to the trained model file.
        :type model_path: str
        """
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the specified input path as a DataFrame.

        This function reads data from the provided input path, assuming it's in CSV format, and returns a pandas DataFrame.
        
        :return: The DataFrame containing the loaded data.
        :rtype: pd.DataFrame
        """
        try:
            logging.info("Loading data from: {}".format(self.input_path))
            pandas_df = pd.read_csv(self.input_path)
            return pandas_df
        except FileNotFoundError:
            logging.error("File not found: {}".format(self.input_path))
            return pd.DataFrame()
        except UnicodeDecodeError as ude:
            logging.error("Unicode decoding error: {}".format(str(ude)))
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            logging.warning("Empty CSV file: {}".format(self.input_path))
            return pd.DataFrame()
        except pd.errors.ParserError as pe:
            logging.error("CSV parsing error: {}".format(str(pe)))
            return pd.DataFrame()
        except PermissionError as perm_error:
            logging.error("Permission error: {}".format(str(perm_error)))
            return pd.DataFrame()
        except OSError as os_error:
            logging.error("OS error: {}".format(str(os_error)))
            return pd.DataFrame()
        except Exception as e:
            logging.error("An error occurred while loading data: {}".format(str(e)))
            return pd.DataFrame()
        
    def load_model(self) -> None:
        """
        Load the trained model from the specified model path.

        This function loads the trained model from the provided model path and stores it in the instance variable 'model'.
        Note: This function uses joblib for model deserialization, but replace 'joblib.load' with the correct function from
        the corresponding library if a different library is used.

        :return: None
        """
        try:
            logging.info("Loading model from: {}".format(self.model_path))
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            logging.error("File not found: {}".format(self.model_path))
        except IsADirectoryError:
            logging.error("The provided model path is a directory: {}".format(self.model_path))
        except PermissionError as perm_error:
            logging.error("Permission error: {}".format(str(perm_error)))
        except OSError as os_error:
            logging.error("OS error: {}".format(str(os_error)))
        except Exception as e:
            logging.error("An error occurred while loading the model: {}".format(str(e))) 

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on the provided data using the trained model.

        This function takes a DataFrame 'data' containing input features, and makes predictions using the trained model.
        The function drops the columns 'Item_Identifier', 'Item_Outlet_Sales', and 'Set' from the input data as they are
        not needed for prediction. Then, it calls the 'predict' method of the trained model and returns the predicted values
        as a pandas DataFrame.

        :param data: The DataFrame containing input features for prediction.
        :type data: pd.DataFrame
        :return: The DataFrame containing the predicted values.
        :rtype: pd.DataFrame
        """
        try:
            logging.info("Making predictions on provided data")
            columns_to_drop = ['Unnamed: 0', 'Outlet_Identifier', 'Item_Identifier', 'Item_Outlet_Sales', 'Item_Outlet_Sales', 'Set']
            data_modified = data.drop(columns=columns_to_drop)
            new_data = self.model.predict(data_modified)
            return new_data
        except ValueError as ve:
            logging.error("ValueError occurred while making predictions: {}".format(str(ve)))
        except Exception as e:
            logging.error("An error occurred while making predictions: {}".format(str(e))) 
        return pd.DataFrame()

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Write the predicted data to a CSV file.

        This function takes the DataFrame 'predicted_data', which contains the predictions, and writes it to a CSV file
        named 'predictions.csv' in the specified output path.

        :param predicted_data: The DataFrame containing the predictions to be written.
        :type predicted_data: pd.DataFrame
        :return: None
        """
        try:
            logging.info("Writing predictions to: {}".format(self.output_path))
            df_predicted_data = pd.DataFrame(
                predicted_data,
                columns=['Prediction']
            )
            df_predicted_data.to_csv(self.output_path + '/predictions.csv')
        except FileNotFoundError:
            logging.error("Output directory not found: {}".format(self.output_path))
        except PermissionError as perm_error:
            logging.error("Permission error: {}".format(str(perm_error)))
        except Exception as e:
            logging.error("An error occurred while writing predictions: {}".format(str(e)))

    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    
    pipeline = MakePredictionPipeline(input_path = '../data/dataframe.csv',
                                      output_path = '../predictions',
                                      model_path = '../model/trained_model.pkl')
    pipeline.run()  