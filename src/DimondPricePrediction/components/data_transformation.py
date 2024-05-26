import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.DimondPricePrediction.utils.utlis import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifaccts','preprocessor.pk1')

class DataTransformation:
    def __init__(self):
        self._data_transformation_config=DataTransformationConfig()


    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')

            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth','table', 'x','y','z']
            
            
            cut_categories = ['Premium', 'Very Good', 'Ideal', 'Good', 'Fair']
            color_categories =['F', 'J', 'G', 'E', 'D', 'H', 'I']
            clarity_categories =['VS2', 'SI2', 'VS1', 'SI1', 'IF', 'VVS2', 'VVS1', 'I1']

            logging.info("Pipeline Initiated")


            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("impter",SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                    ]
            )


            preprocessor=ColumnTransformer([
                ('num_pipline',num_pipeline,numerical_cols),
                ('cat_pipline',cat_pipeline,categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)

    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data complete")
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")

            preprocessing_obj = self.get_data_transformation()

            target_column_name ='price'
            drop_columns = [target_column_name,id]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transformation(input_feature_train_df)

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('applying preprocessor object on training and testing datasets.')

            train_arr = np.c_[input_feature_test_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("preprocessor pickle file saved")

            return(
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)