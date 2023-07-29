"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
import joblib
import logging

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
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
        except Exception as e:
            logging.error("An error occurred while loading data: {}".format(str(e)))  # noqa E501
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
            self.model = joblib.load(self.model_path)  # library  # noqa E501
        except FileNotFoundError:
            logging.error("File not found: {}".format(self.model_path))
        except Exception as e:
            logging.error("An error occurred while loading the model: {}".format(str(e)))  # noqa E501

        return None

    def make_predictions(self, data: DataFrame) -> pd.DataFrame:
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
            data_modified = data.drop(['Item_Identifier', 'Item_Outlet_Sales', 'Item_Outlet_Sales', 'Set'], axis=1, inplace=True)
            new_data = self.model.predict(data_modified)
            return new_data
        except Exception as e:
            logging.error("An error occurred while making predictions: {}".format(str(e)))  # noqa E501
            return pd.DataFrame()

    def write_predictions(self, predicted_data: DataFrame) -> None:
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
        except Exception as e:
            logging.error("An error occurred while writing predictions: {}".format(str(e)))  # noqa E501

        return None


    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    
    pipeline = MakePredictionPipeline(input_path = './data/dataframe.csv',
                                      output_path = './predict',
                                      model_path = './model/trained_model.pkl')
    pipeline.run()  