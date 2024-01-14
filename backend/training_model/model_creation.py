
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os 
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler,OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.compose import ColumnTransformer
from configure_model import  ModelConfiguration
WEIGHT_FOR_FEELS_LIKE = 2.0
WEIGHT_FOR_HUMIDITY = 2.0
    
class ModelCreation(ModelConfiguration):
    
 
    def extract_data_by_date(self, transformed_array, original_dataframe, date_from, date_to):
        # Ensure datetime column is in datetime format
        original_dataframe['datetime'] = pd.to_datetime(original_dataframe['datetime'])
        
        # Find the indices of rows in the original DataFrame that fall within the specified date range
        mask = (original_dataframe['datetime'] >= date_from) & (original_dataframe['datetime'] <= date_to)
        selected_indices = np.where(mask)[0]
        
        # Extract the corresponding rows from the transformed array
        extracted_data = transformed_array[selected_indices]
        
        return extracted_data
        
    def extract_data_by_date_range(self, transformed_array, original_dataframe, start_date, num_days):
        # Ensure datetime column is in datetime format
        original_dataframe['datetime'] = pd.to_datetime(original_dataframe['datetime'])
        
        # Calculate the end date based on the start date and number of days
        end_date = pd.to_datetime(start_date) + pd.DateOffset(days=num_days)
        
        # Find the indices of rows in the original DataFrame that fall within the specified date range
        mask = (original_dataframe['datetime'] >= start_date) & (original_dataframe['datetime'] < end_date)
        selected_indices = np.where(mask)[0]
        
        # Extract the corresponding rows from the transformed array
        extracted_data = transformed_array[selected_indices]
        datetime_column = original_dataframe.loc[mask, 'datetime']
    
        return datetime_column, extracted_data

        # return extracted_data
    
    def fit_transform_pipeline(self, df_after_dropping_columns):
        # print(df_after_dropping_columns.columns.tolist())
        df_after_dropping_columns['conditions'] = df_after_dropping_columns['conditions'].astype(str)
        imputer = SimpleImputer(strategy="median")
        df_num = df_after_dropping_columns.drop("conditions", axis=1)

        zero_columns = df_num.columns[(df_num == 0).all()]
        print("Columns with only zero values:", zero_columns)
        df_num = df_num.drop(zero_columns, axis=1)

        imputer.fit(df_num)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
            ])
        df_num_tr = num_pipeline.fit_transform(df_num)

        num_attribs = list(df_num)
        cat_attribs = ["conditions"]
        cat_pipeline = Pipeline([
            ('onehot', OneHotEncoder())
        ])
        full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
        ])

        num_prepared_matrix = full_pipeline.fit_transform(df_after_dropping_columns)   #df_num_tr 
        joblib.dump(full_pipeline, 'preprocessing_pipeline.joblib')
        return num_prepared_matrix
        
        
        
        
        # num_pipeline = Pipeline([
        #     ('imputer', SimpleImputer(strategy="median")),
        #     ('std_scaler', StandardScaler()),
        #     ])
        # df_num_transformed = num_pipeline.fit_transform(df_num)
        # return df_num_transformed


    def save_csv_prepared(self, df_num):
            
        self.num_prepared = self.fit_transform_pipeline(df_num) #full_pipeline.fit_transform(df_without_date_name)   
        np.savetxt("probaproba.csv", self.num_prepared, delimiter=',')
        return self.num_prepared

    # def set_model_weights(self, X_train, X_test):        
        
    #     weight_for_second_column = WEIGHT_FOR_FEELS_LIKE
    #     weight_for_fourth_column = WEIGHT_FOR_HUMIDITY

    #     feature_weights = np.ones(X_train.shape[1])

    #     feature_weights[1] = weight_for_second_column
    #     feature_weights[3] = weight_for_fourth_column

    #     X_train_weighted = X_train * feature_weights
    #     X_test_weighted = X_test * feature_weights
        
    #     return X_train_weighted, X_test_weighted

    def create_model(self, X_train):
        model = Sequential([
            Dense(self.number_of_neurons_in_first_and_third_hidden_layer, input_dim=X_train.shape[1] , activation=self.activation_relu),
            Dense(self.number_of_neurons_in_second_hidden_layer, activation=self.activation_tanh),
            Dense(self.number_of_neurons_in_first_and_third_hidden_layer, activation=self.activation_leaky_relu),
            Dense(self.number_of_neurons_in_fourth_hidden_layer, activation=self.activation_linear)
        ])
        return model
    
    def compile_fit_predict(self, X_train, X_test, y_train):
                
        self.model = self.create_model(X_train)
        self.model.compile(optimizer = Adam(learning_rate=self.learning_rate_adam),loss=self.cost_function)
        
        
        self.model.summary()

        scaler = MinMaxScaler()

        # Fit the scaler on X_train and transform both X_train and X_test
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        self.model.fit(X_train, y_train, epochs=self.epoch_number, batch_size=self.batch_size_number, verbose=self.verbose)
        trainPredict = self.model.predict(X_train)
        testPredict = self.model.predict(X_test)
        return trainPredict, testPredict
        
        

    


