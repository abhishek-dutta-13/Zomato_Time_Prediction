import os # We use os to create path...
import sys
from src.logger import logger
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder # Ordinal Encoding for categorical variables
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer #Group everything together
from dataclasses import dataclass
from src.utils import *


@dataclass

class DataTransformationConfig:
    #Get Directory path of the current script:
    current_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    #Define artifacts path
    artifact_path = os.path.join(current_directory, "artifacts")

    preprocessor_obj_file_path = os.path.join(artifact_path, "preprocessor_obj.pkl")

class DataTransformation:
    
        def __init__(self) :
            self.data_transformation_config = DataTransformationConfig()

        def get_data_transformation_object(self, categories, one_hot_cols, ordinal_cols, num_cols):
            try:
                # Independent numerical columns
                num_cols_pipe = [col for col in num_cols if col != "Time_taken (min)"]

                # Define pipelines for categorical and numeric data
                categorical_onehot_pipeline = Pipeline([
                    
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(sparse_output=False)),
                    ('scaler', StandardScaler())
                ])

                categorical_ordinal_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', OrdinalEncoder(categories=categories)),
                    ('scaler', StandardScaler())
                ])

                numerical_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])

                # Combine pipelines in a ColumnTransformer
                preprocessor = ColumnTransformer(transformers=[
                    ('cat_one_hot', categorical_onehot_pipeline, one_hot_cols),
                    ('cat_ordinal', categorical_ordinal_pipeline, ordinal_cols),
                    ('num', numerical_pipeline, num_cols_pipe)
                ])

                logger.info("Pipeline methods creation ends!!!")
                return preprocessor
            
            except Exception as e:
                logger.error("Error in get_data_transformation_object")
                raise CustomException(e, sys)



        def inititate_data_transformation(self, train_path, test_path):
            """
            This function is used to initiate the data transformation process.
            """
            try:
                 
                logger.info("Initiating data transformation process...")
                
                df_train = pd.read_csv(train_path)
                df_test = pd.read_csv(test_path)
                logger.info("Data loaded successfully")

                # Droping columns and giving the target column:
                target_column = "Time_taken (min)"
                drop_cols = ['SerialNo', 'ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked' ]
                df_train.drop(labels=drop_cols, axis=1, inplace=True)
                df_test.drop(labels=drop_cols, axis=1, inplace=True)

                #Calculating distance:
                df_train["distance"] = df_train.apply(lambda row: cal_distance(row['Restaurant_latitude'], row['Restaurant_longitude'], row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)
                df_train.drop(labels=[ 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude'], axis=1, inplace=True)

                #Calculating distance:
                df_test["distance"] = df_test.apply(lambda row: cal_distance(row['Restaurant_latitude'], row['Restaurant_longitude'], row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)
                df_test.drop(labels=[ 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude'], axis=1, inplace=True)

                #Replacing 0 with 1.
                df_train["Vehicle_condition"] = df_train["Vehicle_condition"].replace(0,1)
                df_train["multiple_deliveries"] = df_train["multiple_deliveries"].replace(0.0,1.0)
                df_test["Vehicle_condition"] = df_test["Vehicle_condition"].replace(0,1)
                df_test["multiple_deliveries"] = df_test["multiple_deliveries"].replace(0.0,1.0)

                # Calling Feature Classifier for training data:
                feature_classifier_obj = FeatureClassifier(df_train,target_column)
                one_hot_cols, ordinal_cols, num_cols, ordinal_columns_mapping = feature_classifier_obj.ordinal_onehot_numerical_divide()
                logger.info("Categorical columns  and numerical columns divided successfully")

                df_train = fill_empty_with_mode(df_train,one_hot_cols)
                df_train = fill_empty_with_mode(df_train,ordinal_cols)
                logger.info("Empty values filled with mode successfully")

                df_test = fill_empty_with_mode(df_test,one_hot_cols)
                df_test = fill_empty_with_mode(df_test,ordinal_cols)
                logger.info("Empty values filled with mode successfully")

                # Outlier detection:
                df_train = outlier_removal(df_train, num_cols)
                logger.info("Outliers removed!!!")

                # Outlier detection:
                df_test = outlier_removal(df_test, num_cols)
                logger.info("Outliers removed!!!")

                # Listing all the categories:
                categories = []
                for key, value in ordinal_columns_mapping.items():
                    categories.append(value)
                logger.info("Categories created successfully!!!")
                preprocessor_obj = self.get_data_transformation_object(categories, one_hot_cols, ordinal_cols, num_cols)
                
                # Segregation of input and target feature:

                X_train = df_train.drop(labels=target_column, axis=1)
                y_train = df_train[target_column]

                X_test = df_test.drop(labels=target_column, axis=1)
                y_test = df_test[target_column]

                logger.info("Input and target feature segregated successfully!!!")


                #Transformation using preprocessing object:
                X_train_arr = preprocessor_obj.fit_transform(X_train)
                X_test_arr = preprocessor_obj.transform(X_test)
                logger.info("Preprocessing done successfully!!!")

                train_arr = np.c_(X_train_arr, np.array(y_train))
                test_arr = np.c_(X_test_arr, np.array(y_test))

                logger.info("Data transformation done successfully!!!")

                save_objects(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessor_obj
                )
                logger.info(f"Saved the Pickle file preprocessing object to: {self.data_transformation_config.preprocessor_obj_file_path}" )
                
                return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

            except Exception as e:
                logger.error("Error occured while initiating the data transformation process: {}".format(e))
                raise CustomException("Error occured while initiating the data transformation process: {}".format(e, sys))


            










