"""
feature_engineering.py

This module contains functions to perform feature engineering on a DataFrame.

AUTHOR: Karen Raczkowski
FECHA: 18/8/23
"""

# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import stats

class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        """
        Initialize the DataProcessor object.
        
        :param input_path: The path to the directory containing the input data files.
        :type input_path: str
        :param output_path: The path to the directory where the processed data will be saved.
        :type output_path: str
        """
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Read and combine data from 'Train_BigMart.csv' and 'Test_BigMart.csv' into a DataFrame.

        :return: The combined DataFrame with an additional 'Set' column to distinguish between train and test sets.
        :rtype: pd.DataFrame
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

            pandas_df = pd.concat([data_train, data_test], ignore_index=True, sort=False)
            print(pandas_df.head(20))

        except Exception as error: 
            print("An exception ocurred: ", type(error).__name__,"-", error) 

        return pandas_df

    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data transformation on the input DataFrame.

        :param df: The input DataFrame to be transformed.
        :type df: pd.DataFrame
        :return: The transformed DataFrame.
        :rtype: pd.DataFrame
        """
        
        # df.describe()

        # Convert the 'Outlet_Establishment_Year' column to represent years since establishment.
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']

        # Unify labels for "Item_Fat_Content"
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
        
        # Verification of unified labels
        set(df['Item_Fat_Content'])

        # CLEANING: Missing data of Item_Weight variables
        productos = list(df[df['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for producto in productos:
            moda = (df[df['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
            df.loc[df['Item_Identifier'] == producto, 'Item_Weight'] = moda

        # CLEANING: Missing data on Outlet_Size variables
        outlets = list(df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        # New category item fat
        df.loc[df['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        # Generating categories for 'Item_Type'
        df['Item_Type'] = df['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
                                                       'Seafood': 'Meats', 'Meat': 'Meats','Baking Goods': 'Processed Foods', 
                                                       'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
                                                       'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
                                                       'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})
       
        #Codificación de los precios de productos
        df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels = [1, 2, 3, 4])

        #Codificación de variables ordinales
        data = df.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()        
        
        data['Outlet_Size'] = data['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        data['Outlet_Location_Type'] = data['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) 
        print(data.head(5))

        #Codificación de variables nominales
        df_transformed = pd.get_dummies(data, columns=['Outlet_Type'], dtype=int)
        df_transformed.info()
                
        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Write the prepared DataFrame to a CSV file.

        :param transformed_dataframe: The DataFrame that has been prepared and needs to be saved.
        :type transformed_dataframe: pd.DataFrame
        """
        try:
            name_file = 'dataframe.csv'
            output_file = os.path.join(self.output_path, name_file)
            transformed_dataframe.to_csv(output_file, index=False) # Set index=False to exclude index column from CSV.
        except Exception as error: 
            print("An exception ocurred: ", type(error).__name__,"-", error) 
            
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = './data/',
                               output_path = './data/').run()