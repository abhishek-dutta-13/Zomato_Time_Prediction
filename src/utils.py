import pandas as pd
import numpy as np
import pymysql
from src.exception import CustomException
from src.logger import logger
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
import os
import pickle


def mysql_connection():
    # Define the connection parameters:
    logger.info("Connecting to MySQL...")
    try:
        host = "localhost"
        user = "root"
        password = "Munaidi@1415"
        db = 'zomato'

        # Establish connection:

        conn = pymysql.connect(host = host, 
                            user = user,
                            password = password, 
                            database = db, 
                            cursorclass=pymysql.cursors.DictCursor)
        logger.info("MySQL connection established")
        return conn, conn.cursor()
    except Exception as e:
        logger.error("Error connecting to MySQL: {}".format(e))
        raise CustomException("Error connecting to MySQL: {}".format(e))
    # finally:
    #     conn.close()
    #     logger.info("MySQL connection closed")


def cal_distance(source_lat, source_long, destination_lat, destination_long):
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1 = abs(radians(source_lat))
    lon1 = abs(radians(source_long))
    lat2 = abs(radians(destination_lat))
    lon2 = abs(radians(destination_long))
    
    # Calculate the change in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def outlier_removal(df, num_cols):
    for column in num_cols:
        upper_limit = df[column].mean() + 2 * df[column].std()
        lower_limit = df[column].mean() - 2 * df[column].std()
        df = df[(df[column] < upper_limit) & (df[column] > lower_limit)]
    return df

def fill_empty_with_mode(df, cat_cols):
    for i in cat_cols:
        if (df[i] == '').any():
            mode_value = df[i][df[i]!=""].mode().iloc[0]
            df[i] = df[i].replace('',mode_value )
    return df

def random_search_cv(model, X_train, y_train,params):
    random_cv = RandomizedSearchCV(model, param_distributions=params, scoring="r2", cv = 5, verbose=0 )
    random_cv.fit(X_train, y_train)
    return random_cv, random_cv.best_params_, random_cv.best_score_


class FeatureClassifier:
    def __init__(self,df, target_column):
        self.df = df
        self.target_column = target_column
    
    def get_ordinal_columns_mapping(self,columns):
        """
        This function is used to get the mapping of ordinal columns.
        Each key is named as 'ColumnName_Map' and contains the unique values for that column.
        """
        columns_mapping = {}
        
        for col in columns:
            sorted_groups = self.df.groupby(col)[self.target_column].mean().sort_values().index.tolist()
            key_name = f"{col}"
            columns_mapping[key_name] = sorted_groups
        
        return columns_mapping
        

        
    def ordinal_onehot_numerical_divide(self):
        """
        This function is used to divide the categorical into ordinal and one-hot columns and numerical columns.
        """
        one_hot_cols = []
        ordinal_cols = []
        num_cols = []
        #Overall mean
        mean = self.df[self.target_column].mean()
        thereshold_percentage = 0.1
        threshold_value = mean * thereshold_percentage
        try:
            for column in self.df.columns:
                if column != self.target_column and self.df[column].dtype == 'object':
                    df_column = self.df[[column, self.target_column]].groupby(column).mean().reset_index()
                    standard_dev = df_column[self.target_column].std()
                    if standard_dev > threshold_value:
                        ordinal_cols.append(column)
                    else:
                        one_hot_cols.append(column)
                else:
                    num_cols.append(column)
            
            logger.info("Outliers removed!!!")

            #Get Mappingsd for ordinal columns:
            ordinal_columns_mapping = self.get_ordinal_columns_mapping(ordinal_cols)
            one_hot_column_mapping = self.get_ordinal_columns_mapping(one_hot_cols)
            return (one_hot_cols, ordinal_cols, num_cols, ordinal_columns_mapping, one_hot_column_mapping)
                 

        except Exception as e:
            print(e)
            raise CustomException("Error in feature_classifier.ordinal_onehot_numerical_divide: {}".format(e))

def save_objects(file_path, obj):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info("Object saved successfully")
    except Exception as e:
        logger.error("Error in save_objects: {}".format(e))
        raise CustomException("Error in save_objects: {}".format(e))
    

def load_obj(file_path):
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info("Object loaded successfully")
        return obj
    except Exception as e:
        logger.error("Error in load_obj: {}".format(e))
        raise CustomException("Error in load_obj: {}".format(e))
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    report = {}
    try:
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = test_model_score
            logger.info(f"Model: {list(models.keys())[i]}, R2 score: {test_model_score}")
        logger.info("Model evaluation complete")
        return report

    except Exception as e:
        logger.error("Error in evaluate_model: {}".format(e))
        raise CustomException("Error in evaluate_model: {}".format(e))