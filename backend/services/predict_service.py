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
        print(df.shape)

        prepared_data = self.model_creation.fit_transform_pipeline(df_)
        
        print(df.shape)
        print(prepared_data.shape)
        extracted_data = self.model_creation.extract_data_by_date_range(prepared_data, df, date_from, num_of_days)
        # extracted_data = self.model_creation.extract_data_by_date(prepared_data, df, '2021-08-01', '2021-08-07')
        
        results = loaded_model.predict(extracted_data)

        return results


    def extract_data_by_date_range( transformed_array, original_dataframe, start_date, num_days):
        # Ensure datetime column is in datetime format
        original_dataframe['datetime'] = pd.to_datetime(original_dataframe['datetime'])
        
        # Calculate the end date based on the start date and number of days
        end_date = pd.to_datetime(start_date) + pd.DateOffset(days=num_days)
        
        # Find the indices of rows in the original DataFrame that fall within the specified date range
        mask = (original_dataframe['datetime'] >= start_date) & (original_dataframe['datetime'] <= end_date)
        selected_indices = np.where(mask)[0]
        
        # Extract the corresponding rows from the transformed array
        extracted_data = transformed_array[selected_indices]
        
        return extracted_data 