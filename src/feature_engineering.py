"""
feature_engineering.py

This module contains functions to perform feature engineering on a DataFrame.

AUTHOR: Karen Raczkowski
DATE: 18/8/23
"""

# Imports
import os
import pandas as pd


class FeatureEngineeringPipeline:
    """
    A class for performing feature engineering on BigMart sales data.

    This class provides methods to read data, perform data transformation, 
    and write the prepared data to a CSV file.

    Args:
        input_path (str): The path to the directory containing the input data files.
        output_path (str): The path to the directory where the processed data will be saved.

    Attributes:
        input_path (str): The path to the directory containing the input data files.
        output_path (str): The path to the directory where the processed data will be saved.

    Methods:
        read_data(): Read and combine data from 'Train_BigMart.csv' and 'Test_BigMart.csv'.
        data_transformation(df: pd.DataFrame) -> pd.DataFrame: 
        Perform data transformation on the provided DataFrame.
        write_prepared_data(transformed_dataframe: pd.DataFrame) -> None: 
        Write the prepared DataFrame to a CSV file.
        run(): Execute the complete feature engineering pipeline by reading, 
        transforming, and writing data.
    """

    def __init__(self, input_path, output_path):
        """
        Initialize the FeatureEngineeringPipeline object.

        Args:
        input_path (str): The path to the directory containing the input data files.
        output_path (str): The path to the directory where the processed data will be saved.
        """
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Read and combine data from 'Train_BigMart.csv' and 'Test_BigMart.csv' into a DataFrame.

        Returns:
        pd.DataFrame: The combined DataFrame with an additional 'Set' column 
        to distinguish between train and test sets.
        """
        try:
            train_file = 'Train_BigMart.csv'
            train_data = os.path.join(self.input_path, train_file)
            data_train = pd.read_csv(train_data)
            data_train['Set'] = 'train'

            test_file = 'Test_BigMart.csv'
            test_data = os.path.join(self.input_path, test_file)
            data_test = pd.read_csv(test_data)
            data_test['Set'] = 'test'

            pandas_df = pd.concat([data_train, data_test],
                                  ignore_index=True, sort=False)

        except FileNotFoundError:
            print("Error: File not found or directory does not exist.")
        except PermissionError:
            print("Error: Permission error while accessing the file.")
        except OSError as error:
            print("Error: OS error -", error)
        except pd.errors.ParserError as error:
            print("Error: Parser error -", error)
        except MemoryError:
            print("Error: Memory error - file is too large to load.")
        except Exception as error:
            print("An unexpected error occurred:",
                  type(error).__name__, "-", error)

        return pandas_df

    def data_transformation(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data transformation on the provided DataFrame.

        Args:
        data_frame (pd.DataFrame): Input DataFrame containing BigMart sales data.

        Returns:
        pd.DataFrame: Transformed DataFrame with cleaned and processed features.
        """
        
        # Convert the 'Outlet_Establishment_Year' column to represent years since establishment.
        data_frame['Outlet_Establishment_Year'] = 2020 - \
            data_frame['Outlet_Establishment_Year']

        # Standardize Item_Fat_Content values
        data_frame['Item_Fat_Content'] = data_frame['Item_Fat_Content'].replace(
            {'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        # Impute missing Item_Weight values using mode for each product
        productos = list(data_frame[data_frame['Item_Weight'].isnull()]
                         ['Item_Identifier'].unique())
        for producto in productos:
            moda = (data_frame[data_frame['Item_Identifier'] == producto]
                    [['Item_Weight']]).mode().iloc[0, 0]
            data_frame.loc[data_frame['Item_Identifier']
                           == producto, 'Item_Weight'] = moda

        # Impute missing Outlet_Size values with 'Small'
        outlets = list(data_frame[data_frame['Outlet_Size'].isnull()]
                       ['Outlet_Identifier'].unique())
        for outlet in outlets:
            data_frame.loc[data_frame['Outlet_Identifier']
                           == outlet, 'Outlet_Size'] = 'Small'

        # Update Item_Fat_Content for specific Item_Type
        data_frame.loc[data_frame['Item_Type'] ==
                       'Non perishable', 'Item_Fat_Content'] = 'NA'

        # Group similar Item_Type categories and update them
        data_frame['Item_Type'] = data_frame['Item_Type'].replace({'Others': 'Non perishable',
                                                                   'Health and Hygiene': 'Non perishable',
                                                                   'Household': 'Non perishable',
                                                                   'Seafood': 'Meats', 'Meat': 'Meats',
                                                                   'Baking Goods': 'Processed Foods',
                                                                   'Frozen Foods': 'Processed Foods',
                                                                   'Canned': 'Processed Foods',
                                                                   'Snack Foods': 'Processed Foods',
                                                                   'Breads': 'Starchy Foods',
                                                                   'Breakfast': 'Starchy Foods',
                                                                   'Soft Drinks': 'Drinks',
                                                                   'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

        # Discretize Item_MRP into 4 quantiles
        data_frame['Item_MRP'] = pd.qcut(
            data_frame['Item_MRP'], 4, labels=[1, 2, 3, 4])

        # Create a copy of the DataFrame with certain columns dropped
        data = data_frame.drop(
            columns=['Item_Type', 'Item_Fat_Content']).copy()

        # Convert categorical variables to numerical
        data['Outlet_Size'] = data['Outlet_Size'].replace(
            {'High': 2, 'Medium': 1, 'Small': 0})
        data['Outlet_Location_Type'] = data['Outlet_Location_Type'].replace(
            {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})
        # print(data.head(5))

        # One-hot encode Outlet_Type column
        df_transformed = pd.get_dummies(
            data, columns=['Outlet_Type'], dtype=int)
        df_transformed.info()

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Write the prepared DataFrame to a CSV file.

        Args:
        transformed_dataframe (pd.DataFrame):
        The DataFrame that has been prepared and needs to be saved.
        """
        try:
            name_file = 'dataframe.csv'
            output_file = os.path.join(self.output_path, name_file)
            transformed_dataframe.to_csv(output_file)
        except (IOError, OSError, PermissionError,
                FileNotFoundError, pd.errors.ParserError) as error:
            print("An exception ocurred: ", type(error).__name__, "-", error)
            print("Error writing to file")
        except Exception as error:
            print("An unexpected error occurred:",
                  type(error).__name__, "-", error)

    def run(self):
        """
        Execute the complete feature engineering pipeline by reading, 
        transforming, and writing data.
        """
        data_frame = self.read_data()
        df_transformed = self.data_transformation(data_frame)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path='../data/',
                               output_path='../data/').run()
