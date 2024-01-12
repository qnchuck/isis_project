# from model_training_service import ModelTrainingService
from training_model.model_creation import ModelCreation
from training_model.data_preprocessing_copy import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from db.database import DatabaseHandler

db_handler = DatabaseHandler()

class PredictService:
    def __init__(self):
        self.model_creation = ModelCreation()

    def predict(self, date_from, num_of_days):
        loaded_model = load_model('my_model_2.h5')
        
        # print(date_from)
        df,df_ = db_handler.read_preprocessed_data()

        prepared_data = self.model_creation.fit_transform_pipeline(df_)
        
        date_time_column, extracted_data = self.model_creation.extract_data_by_date_range(prepared_data, df, date_from, num_of_days)
        # extracted_data = self.model_creation.extract_data_by_date(prepared_data, df, '2021-08-01', '2021-08-07')
        
        results = loaded_model.predict(extracted_data)

        results_df = pd.DataFrame({'Load': results.flatten()})
        datetimec= pd.DataFrame({'datetime': date_time_column})
        datetimec = datetimec.reset_index()

        datetimec['datetime'] = pd.to_datetime(datetimec['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        print(datetimec)
        # Concatenate DataFrames
        merged_df = pd.concat([datetimec, results_df], axis=1)
        merged_df = merged_df.drop('index', axis=1)

        # Convert the merged DataFrame to JSON
        json_data = merged_df.to_json(orient='records')
        print(json_data)
        return json_data