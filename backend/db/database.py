from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime, timedelta
# Replace these variables with your actual database connection details
DB_USER = 'postgres'
DB_PASSWORD = 'admin'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'forecast'

class DatabaseHandler():
    def __init__(self):
        self.engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    def read_preprocessed_data(self, start_date, end_date):
        query = f"SELECT * FROM weather_data WHERE datetime BETWEEN '{start_date}' AND '{end_date}'"
        df_with_datetime = pd.read_sql(query, self.engine)
        df_dropped_datetime = df_with_datetime.drop("datetime", axis=1)
        return df_with_datetime, df_dropped_datetime
        
    def read_data(self, table_name):
        query = f'SELECT * FROM {table_name}'
        df = pd.read_sql(query, self.engine)
        return df

    def write_preprocessed_data(self, new_data):
        df = pd.DataFrame(new_data)
        df.to_sql('weather_data', self.engine, if_exists='replace', index=False)
    
    def read_load_data_by_date(self, start_date, end_date):
        query = f"SELECT * FROM load_data WHERE datetime BETWEEN '{start_date}' AND '{end_date}'"

        # Read data from the database into a DataFrame
        df = pd.read_sql(query, self.engine)
        return df
    
    def read_preprocessed_data_for_forecast(self):
        query = f'SELECT * FROM weather_data'
        df_with_datetime = pd.read_sql(query, self.engine)
        df_dropped_datetime = df_with_datetime.drop("datetime", axis=1)
        return df_with_datetime, df_dropped_datetime
    
    def write_load_data(self, new_data):
        df = pd.DataFrame(new_data)
        df.to_sql('load_data', self.engine, if_exists='replace', index=False)

    def read_weather_data(self, start_date_str, num_days):
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

        # Format start_date to a string compatible with SQL query
        start_date_str = start_date.strftime('%Y-%m-%d')

        # Calculate the end_date by adding num_days to start_date
        end_date = start_date + timedelta(days=num_days)
        end_date_str = end_date.strftime('%Y-%m-%d')

        query = f'''
        SELECT * FROM weather_data
        WHERE datetime >= '{start_date_str}' AND datetime < '{end_date_str}'
        '''
        df_with_datetime = pd.read_sql(query, self.engine)
        df_dropped_datetime = df_with_datetime.drop("datetime", axis=1)
      
        return df_with_datetime, df_dropped_datetime

    def write_forecast(self, df_load):
        df_load.to_sql('forecast_data', self.engine, if_exists='replace', index=False)
        