from database import DatabaseHandler
from training_model.model_creation import ModelCreation
from tensorflow.keras.models import load_model
import pandas as pd 
import numpy as np
mc = ModelCreation()
db_handler = DatabaseHandler()

# df = db_handler.read_data('weather_data')

# print(df)



df22,df11 = db_handler.read_preprocessed_data()
# print(df22.head())
# print(df11.head())

df3,df33  = db_handler.read_weather_data('2018-01-02',10000)
# print(df3)
# print(df33)



# tf = mc.fit_transform_pipeline(df33)
# loaded_model = load_model('my_model_2.h5')

rows_per_day = 24  # Assuming hourly data
last_seven_days_rows = 7 * rows_per_day


# X_test = tf[-last_seven_days_rows:, :]

# results = loaded_model.predict(X_test)

# print(results)
 
def extract_data_by_date(transformed_array, original_dataframe, date_from, date_to):
        # Ensure datetime column is in datetime format
    original_dataframe['datetime'] = pd.to_datetime(original_dataframe['datetime'])
    
    # Find the indices of rows in the original DataFrame that fall within the specified date range
    mask = (original_dataframe['datetime'] >= date_from) & (original_dataframe['datetime'] <= date_to)
    selected_indices = np.where(mask)[0]
    
    # Extract the corresponding rows from the transformed array
    extracted_data = transformed_array[selected_indices]
    
    return extracted_data


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

 
 
df1,df = db_handler.read_preprocessed_data() #('2018-01-01',10000)   
# df2,d3 = db_handler.read_weather_data('2018-01-01',1000000)   
tf = mc.fit_transform_pipeline(df)
# tf1 = mc.fit_transform_pipeline(d3)
# tf = tf[-last_seven_days_rows:, :]


loaded_model = load_model('my_model_2.h5')
print(loaded_model.summary())

print(tf.shape)
# print(tf1.shape)
# print(d3.shape)
print(df.shape)


extracted_data = extract_data_by_date(tf, df1, '2021-08-01', '2021-08-10')
extracted_data = extract_data_by_date_range(tf,df1, '2021-08-01', 10)
print(extracted_data.shape)
results = loaded_model.predict(extracted_data)
print(results)