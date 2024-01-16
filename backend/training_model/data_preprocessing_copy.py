import pandas as pd
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import os 
from db.database import DatabaseHandler
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sqlalchemy import create_engine
import numpy as np

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

db_handler = DatabaseHandler()

folder_path_weather = "/home/qnchuck/Desktop/isis/backend/uploads/weather"  
file_pattern_weather = "New York City,*.csv"

excel_file_path = '/home/qnchuck/Desktop/csv_data/US Holidays 2018-2021.xlsx'
root_folder_load = "/home/qnchuck/Desktop/isis/backend/uploads/load"  

# from database.database import DatabaseHandler
# db_handler = DatabaseHandler()

def read_weather_data_from_folder(folder_path, file_pattern, date_from, date_to):
    files = glob.glob(f"{folder_path}/{file_pattern}")

    # Read files into a pandas DataFrame and remove nulls
    dfs = [pd.read_csv(file) for file in files]
    
    concated_dfs = concat_dataframes_into_df(dfs)
    removed_nulls_dfs = remove_nulls_from_dataframes(concated_dfs)
    
    # Filter data based on date range
    removed_nulls_dfs = filter_data_by_date_range(removed_nulls_dfs, 'datetime', date_from, date_to)
    
    return removed_nulls_dfs

def read_load_data_from_folder(root_folder, date_from, date_to):
    all_dfs = []

    # for folder_name in os.listdir(root_folder):
    #     if folder_name.endswith("pal_csv") and os.path.isdir(os.path.join(root_folder, folder_name)):
    #         folder_path = os.path.join(root_folder, folder_name)

    for filename in os.listdir(root_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(root_folder, filename)
            # Read CSV into DataFrame
            df = pd.read_csv(file_path)

            df = df[df['Name'] == 'N.Y.C.']

            # Append DataFrame to the list
            all_dfs.append(df)

    concated_dfs = concat_dataframes_into_df(all_dfs)
    removed_nulls_df = remove_nulls_from_dataframes(concated_dfs)
    
    # Filter data based on date range
    removed_nulls_df = filter_data_by_date_range(removed_nulls_df, 'Time Stamp', date_from, date_to)

    return removed_nulls_df

def filter_data_by_date_range(df, date_column, date_from, date_to):  
    print(type(date_from))

    date_from = np.datetime64(date_from)
    date_to = np.datetime64(date_to)
    df[date_column] = pd.to_datetime(df[date_column])
    print(type(df[date_column][3]))
    print(type(date_from))
    df = df[(df[date_column] >= date_from) & (df[date_column] <= date_to)]
    return df

# Your other existing methods...

def merge_load_and_weather_data(date_from_train, date_to_train):
    
    dfs_weather = read_and_modify_weather_data(date_from_train, date_to_train)
    dfs_load = read_and_modify_load_data(date_from_train, date_to_train)
    print(date_to_train)
    df_weather_reset = dfs_weather.reset_index()
    # Rename the "index" column to "datetime"
    df_weather_reset = df_weather_reset.rename(columns={'index': 'datetime'})

    df_load_reset = dfs_load.reset_index()
    # Rename the "index" column to "datetime"
    df_load_reset = df_load_reset.rename(columns={'index': 'Time Stamp'})

    df_merged = pd.merge(df_weather_reset, df_load_reset, left_on='datetime', right_on='Time Stamp', how='inner')

    # Drop redundant datetime column from df_load
    df_merged = df_merged.drop(['Time Stamp', 'Time Zone', 'Name', 'PTID', 'level_0_y'], axis=1)

    # Filter merged data for test date range
    # df_merged_test = filter_data_by_date_range(df_merged, 'datetime', date_from_test, date_to_test)

    # df_merged_test.to_csv('merged_data_test.csv', index=False)
    
    return df_merged







def remove_nulls_from_dataframes(dfs):
    
    dfs = dfs.interpolate(limit_area='inside').fillna(0)
    return dfs

# def read_weather_data_from_folder(folder_path, file_pattern):
#     files = glob.glob(f"{folder_path}/{file_pattern}")

#     #Read files into a pandas DataFrame and remove nulls
#     dfs = [pd.read_csv(file) for file in files]
    
#     concated_dfs = concat_dataframes_into_df(dfs)
#     removed_nulls_dfs = remove_nulls_from_dataframes(concated_dfs)
    
#     return removed_nulls_dfs

# def read_load_data_from_folder(root_folder):
#     all_dfs = []

#     for folder_name in os.listdir(root_folder):
#         if folder_name.endswith("pal_csv") and os.path.isdir(os.path.join(root_folder, folder_name)):
#             folder_path = os.path.join(root_folder, folder_name)

#             for filename in os.listdir(folder_path):
#                 if filename.endswith(".csv"):
#                     file_path = os.path.join(folder_path, filename)
#                     # Read CSV into DataFrame
#                     df = pd.read_csv(file_path)

#                     df = df[df['Name'] == 'N.Y.C.']

#                     # Append DataFrame to the list
#                     all_dfs.append(df)

#     concated_dfs = concat_dataframes_into_df(all_dfs)
#     removed_nulls_df = remove_nulls_from_dataframes(concated_dfs)
#     return removed_nulls_df

def remove_not_full_hour_values(df,date_column):
    
    df[date_column] = pd.to_datetime(df[date_column])

    df = df[(df[date_column].dt.minute == 0) & (df[date_column].dt.second == 0)]

    df = df.sort_values(by=date_column)
    return df
        
def concat_dataframes_into_df(dfs):
    df = pd.concat(dfs, ignore_index=True)
    return df


def add_missing_dates_into_df(df,date_column):
    dfs_to_add_values = df.set_index(date_column)
    
    # Create a complete date range for the desired time period
    complete_date_range = pd.date_range(start=dfs_to_add_values.index.min(), end=dfs_to_add_values.index.max(), freq='H')
    
    dfs_to_add_values = dfs_to_add_values[~dfs_to_add_values.index.duplicated(keep='first')]

    df_filled = dfs_to_add_values.reindex(complete_date_range)
    df_filled = df_filled.reset_index()
    
    #maybe to move this into merge method, but works well here also
    if date_column=="datetime":
        df_filled['conditions'] = df_filled['conditions'].replace(0, method='ffill') 
        df_filled['conditions'] = df_filled['conditions'].replace(0, method='bfill') # in case there is no previous non zero value

    df_filled = df_filled.interpolate(limit_area='inside').fillna(0)

    return df_filled
    

def read_and_modify_weather_data(datefrom, dateto):
    dfs_weather = read_weather_data_from_folder(folder_path_weather, file_pattern_weather, datefrom, dateto)
    dfs_weather = remove_not_full_hour_values(dfs_weather, 'datetime')
    dfs_weather = add_missing_dates_into_df(dfs_weather,'datetime')
    return dfs_weather

def read_and_modify_load_data(datefrom, dateto):
    dfs_load = read_load_data_from_folder(root_folder_load, datefrom, dateto)
    dfs_load = remove_not_full_hour_values(dfs_load, 'Time Stamp')
    dfs_load = add_missing_dates_into_df(dfs_load, 'Time Stamp')
    return dfs_load


def remove_special_dates(df):
    df_special = pd.read_excel(excel_file_path)
    
    special_dates = df_special['Unnamed: 2']

   
    df = df[~df['datetime'].dt.date.isin(special_dates.dt.date)]
     
    return df

def remove_covid_lockdown_dates_in_new_york(df):
    lockdown_start_date = "2020-01-01"
    lockdown_end_date = "2020-12-31"  # Adjust the end date as needed

    df['datetime'] = pd.to_datetime(df['datetime'])

    df_filtered = df[~((df['datetime'] >= lockdown_start_date) & (df['datetime'] <= lockdown_end_date))]

    return df_filtered


def write_df_to_postgresql(df, table_name, connection_string):
    """
    Write a Pandas DataFrame to a PostgreSQL table.

    Parameters:
    - df: Pandas DataFrame
    - table_name: Name of the table in PostgreSQL
    - connection_string: Connection string for PostgreSQL (e.g., 'postgresql://username:password@localhost:5432/database_name')

    Returns:
    - None
    """
    # Create a database connection
    engine = create_engine(connection_string)

    # Write the DataFrame to the PostgreSQL table
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def encode_conditions_column(df_conditions):
    
    cat_encoder = OneHotEncoder()
    df_conditions = df_conditions.replace(0, method='ffill') 
    df_conditions = df_conditions.replace(0, method='bfill') # in case there is no previous non zero value

    df_cat_1hot = cat_encoder.fit_transform(df_conditions)
    return df_cat_1hot

def imputer_contidions(dataframe):
        
    dataframe['conditions'] = dataframe['conditions'].astype(str)
    imputer = SimpleImputer(strategy="median")
    df_num = dataframe.drop("conditions", axis=1)
    imputer.fit(df_num)
    return imputer
        


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_columns_for_day_type(dataframe_without_day_type):
    
    day_types = {
    0: 1,  # Monday
    1: 2,  # Midweek
    2: 2,  # Midweek
    3: 2,  # Midweek
    4: 3,  # Friday
    5: 4,  # Weekend
    6: 4   # Weekend
    }
    
    dataframe_without_day_type['day_of_week'] = dataframe_without_day_type['datetime'].dt.dayofweek
    dataframe_without_day_type['day_type'] = dataframe_without_day_type['day_of_week'].map(day_types)
    return dataframe_without_day_type

def remove_extreme_ouliers_for_temp(df_with_extreme_values, column_name):
    # Calculate the Z-scores for each value in the column
    z_scores = (df_with_extreme_values[column_name] - df_with_extreme_values[column_name].mean()) / df_with_extreme_values[column_name].std()

    # Define a threshold for Z-scores (e.g., 3 for extreme outliers)
    threshold = 3

    # Create a boolean mask to identify extreme values
    outliers_mask = (abs(z_scores) > threshold)

    # Remove extreme outliers and create a new DataFrame
    df_no_outliers = df_with_extreme_values[~outliers_mask] 

    # Interpolate missing values in the 'temp' column
    df_no_outliers[column_name] = df_no_outliers[column_name].interpolate(limit_area='inside').fillna(0)

    df_with_extreme_values[column_name] = df_no_outliers[column_name].copy()

    return df_with_extreme_values

def remove_unnecessary_columns(dataframe):
    df_with_num_columns = dataframe.drop(['Load','name','level_0_x','precipprob','precip', 'severerisk','uvindex','snowdepth','solarenergy','preciptype','snow','day_of_week',], axis=1) 
    
    df_with_num_columns.to_csv('head_columns.csv', index=False)
    
    df_with_num_columns = df_with_num_columns.set_index('datetime')
    

    return df_with_num_columns

def remove_columns_(dataframe):
    df_with_num_columns = dataframe.drop(['Load','name','level_0_x','precipprob','precip','severerisk','uvindex','snowdepth','solarenergy','preciptype','snow','day_of_week',], axis=1) 
    return df_with_num_columns
def remove_columns_additonal_dates(dataframe):
    df_with_num_columns = dataframe.drop(['name','level_0','precipprob','precip','severerisk','uvindex','snowdepth','solarenergy','preciptype','snow','day_of_week',], axis=1) 
    return df_with_num_columns

def save_into_database(dataframe):
    return True


def calculate_heat_index(temperature, humidity):
    # Coefficients for the Steadman's formula
    c = [-42.379, 2.04901523, 10.14333127, -0.22475541, -6.83783e-03, -5.481717e-02, 1.22874e-03, 8.5282e-04, -1.99e-06]
    
    # Convert temperature to Fahrenheit if it's in Celsius
    if temperature < 80:
        temperature = temperature * 9/5 + 32
    
    # Calculate heat index
    heat_index = c[0] + c[1] * temperature + c[2] * humidity + c[3] * temperature * humidity + \
                  c[4] * temperature**2 + c[5] * humidity**2 + c[6] * temperature**2 * humidity + \
                  c[7] * temperature * humidity**2 + c[8] * temperature**2 * humidity**2
    
    return heat_index

def create_month_type(df_without_month_type):
    # Define a mapping from month to season
    month_to_season = {
        1: 4,  # Winter
        2: 4,  # Winter
        3: 1,  # Spring
        4: 1,  # Spring
        5: 1,  # Spring
        6: 2,  # Summer
        7: 2,  # Summer
        8: 2,  # Summer
        9: 3,  # Fall
        10: 3,  # Fall
        11: 3,  # Fall
        12: 4,  # Winter
    }


    # Create a new 'season' column based on the mapping
    df_without_month_type['season'] = df_without_month_type['datetime'].dt.month.map(month_to_season)
    return df_without_month_type


def create_additional_columns(dataframe):
    
    #Calculating of the temperature one day before
    # dataframe['previous_day_temp'] = dataframe.groupby(dataframe['datetime'].dt.hour)['temp'].shift(24)
    # dataframe['temp_seven_days_before'] = dataframe['temp'].shift(7 * 24)  # Assuming your data is hourly
    dataframe['heat_index'] = dataframe.apply(lambda row: calculate_heat_index(row['temp'], row['humidity']), axis=1)
    
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    dataframe['day_of_week'] = dataframe['datetime'].dt.dayofweek
    dataframe['day_of_year_sin'] = np.sin(2 * np.pi * dataframe['datetime'].dt.dayofyear / 365)
    dataframe['day_of_week_sin'] = np.sin(2 * np.pi * dataframe['datetime'].dt.dayofweek / 7)
    dataframe['hour_sin'] = np.sin(2 * np.pi * dataframe['datetime'].dt.hour / 24)
    
    dataframe['day_of_year_cos'] = np.cos(2 * np.pi * dataframe['datetime'].dt.dayofyear / 365)
    dataframe['day_of_week_cos'] = np.cos(2 * np.pi * dataframe['datetime'].dt.dayofweek / 7)
    dataframe['hour_cos'] = np.cos(2 * np.pi * dataframe['datetime'].dt.hour / 24)

    return dataframe


def create_pipelines(df_after_dropping_columns):
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

    from sklearn.compose import ColumnTransformer
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
    
    return num_prepared_matrix

def drop_unnecessary_columns(dataframe_to_remove_columns):
    
    dataframe_dropped = dataframe_to_remove_columns.drop(['Load','name','level_0_x','precipprob', 'severerisk','uvindex', 'solarenergy','preciptype','snow','day_of_week',], axis=1) 
    dataframe_dropped['conditions'] = dataframe_dropped['conditions'].astype(str)
    
    return dataframe_dropped

def remove_columns_with_zero_values(dataframe):
        
    zero_columns = dataframe.columns[(dataframe == 0).all()]
    print("Columns with only zero values:", zero_columns)
    dataframe = dataframe.drop(zero_columns, axis=1)
    
    return dataframe
    
def do_final_preparations_for_model(date_from_train,date_to_train):
    date_from_train = '2018-01-01'
    date_to_train = '2021-09-20'
    dataframe = merge_load_and_weather_data(date_from_train, date_to_train)
    print('im here')
    try:
        dataframe = remove_extreme_ouliers_for_temp(dataframe, 'temp')
        dataframe = create_additional_columns(dataframe)
        dataframe = remove_special_dates(dataframe)
        # dataframe = remove_covid_lockdown_dates_in_new_york(dataframe)
        dataframe = dataframe.reset_index()


        dataframe = dataframe.sort_values(by='datetime')
        # dataframe = create_columns_for_day_type(dataframe)
        # dataframe = create_month_type(dataframe)
        try:
        # df_with_dates = dataframe.copy()
            prepared_load = dataframe[['Load','datetime']] 
            # print(prepared_load)
        
            db_handler.write_load_data(prepared_load)
        except Exception as e:
            print("EXCEPTION IN WRITE LOAD ",e)
        prepared_dataframe = remove_columns_(dataframe)

        db_handler.write_preprocessed_data(prepared_dataframe)

        df_with_dates, prepared_dataframe___ = db_handler.read_preprocessed_data(date_from_train, date_to_train)
        import traceback
        max_weather_date = db_handler.get_max_dates_from_tables().date()
        date_object_ = datetime.strptime(date_to_train, '%Y-%m-%d').date()
        
        print("sent date:",date_to_train)
        print("weather date from db:",max_weather_date)
        # date_object_db = datetime.strptime(max_weather_date, '%Y-%m-%d').date()
        if date_object_!= max_weather_date:
            from datetime import timedelta
            print("sent date:",date_to_train)
            print("weather date from db:",max_weather_date)
            max_weather_date = max_weather_date + timedelta(days=1)
            dfs_weather = read_and_modify_weather_data(max_weather_date, date_object_)
            df_weather_reset = dfs_weather.reset_index()
            # Rename the "index" column to "datetime"
            df_weather_reset = df_weather_reset.rename(columns={'index': 'datetime'})
            dataframe = df_weather_reset
            dataframe = remove_extreme_ouliers_for_temp(dataframe, 'temp')
            dataframe = create_additional_columns(dataframe)
            dataframe = remove_special_dates(dataframe)
            # dataframe = remove_covid_lockdown_dates_in_new_york(dataframe)
            dataframe = dataframe.reset_index()


            dataframe = dataframe.sort_values(by='datetime')
            # dataframe = create_columns_for_day_type(dataframe)
            # dataframe = create_month_type(dataframe)
            prepared_dataframe_additional = remove_columns_additonal_dates(dataframe)
            # df_to_db = pd.merge(prepared_dataframe, prepared_dataframe_additional, on='datetime', how='outer', suffixes=('_df1', '_df2'))
            df_to_db = pd.concat([prepared_dataframe, prepared_dataframe_additional], ignore_index=True)  # ignore_index resets the index
            # df_to_db = prepared_dataframe + prepared_dataframe_additional
            print("shape : ",prepared_dataframe_additional.shape)
            print(prepared_dataframe.shape)
            print(df_to_db.shape)
            print("df head : ",prepared_dataframe_additional.head(0))
            print(prepared_dataframe.head(0))
            print(df_to_db.head(0))
            db_handler.write_preprocessed_data(df_to_db)
        else:
            db_handler.write_preprocessed_data(prepared_dataframe)

        # if date_to_train != max_weather_date:
        
        return  prepared_dataframe, prepared_load, df_with_dates

    except Exception as e:
        traceback.print_exc()
        print("ERROR: ", e)
        return []
# your_dataframe = merge_load_and_weather_data()

# df_cat = encode_conditions_column(your_dataframe[["conditions"]])

        
# # your_dataframe = remove_special_dates(your_dataframe)
# # your_dataframe = remove_covid_lockdown_dates_in_new_york(your_dataframe)
# # y = your_dataframe['Load'] 
# # your_dataframe = your_dataframe.sort_values(by='datetime')



# # Calculate the Z-scores for each value in the column
# z_scores = (your_dataframe['temp'] - your_dataframe['temp'].mean()) / your_dataframe['temp'].std()

# # Define a threshold for Z-scores (e.g., 3 for extreme outliers)
# threshold = 3

# # Create a boolean mask to identify extreme values
# outliers_mask = (abs(z_scores) > threshold)


# # Remove extreme outliers and create a new DataFrame
# df_no_outliers = your_dataframe[~outliers_mask] 

# # Interpolate missing values in the 'temp' column
# df_no_outliers['temp'] = df_no_outliers['temp'].interpolate(limit_area='inside').fillna(0)


# your_dataframe['temp'] = df_no_outliers['temp'].copy()


# #Calculating of the temperature one day before
# your_dataframe['previous_day_temp'] = your_dataframe.groupby(your_dataframe['datetime'].dt.hour)['temp'].shift(24)
# your_dataframe['temp_seven_days_before'] = your_dataframe['temp'].shift(7 * 24)  # Assuming your data is hourly
# your_dataframe['datetime'] = pd.to_datetime(your_dataframe['datetime'])

# your_dataframe['day_of_week'] = your_dataframe['datetime'].dt.dayofweek
# your_dataframe['day_of_year_sin'] = np.sin(2 * np.pi * your_dataframe['datetime'].dt.dayofyear / 365.25)
# your_dataframe['day_of_week_sin'] = np.sin(2 * np.pi * your_dataframe['datetime'].dt.dayofweek / 7)

# # Define a mapping from month to season
# month_to_season = {
#     1: 0,  # Winter
#     2: 0,  # Winter
#     3: 1,  # Spring
#     4: 1,  # Spring
#     5: 1,  # Spring
#     6: 2,  # Summer
#     7: 2,  # Summer
#     8: 2,  # Summer
#     9: 3,  # Fall
#     10: 3,  # Fall
#     11: 3,  # Fall
#     12: 0,  # Winter
# }


# # Create a new 'season' column based on the mapping
# your_dataframe['season'] = your_dataframe['datetime'].dt.month.map(month_to_season)


# #naknadno dodato

# your_dataframe = remove_special_dates(your_dataframe)
# your_dataframe = remove_covid_lockdown_dates_in_new_york(your_dataframe)
# y = your_dataframe['Load'] 
# your_dataframe = your_dataframe.sort_values(by='datetime')
# #naknadno dodato

# # your_dataframe['hour_of_day_sin'] = np.sin(2 * np.pi * your_dataframe['datetime'].dt.hour / 24)
# # Map days to types as integers

# day_types = {
#     0: 1,  # Monday
#     1: 2,  # Midweek
#     2: 2,  # Midweek
#     3: 2,  # Midweek
#     4: 3,  # Friday
#     5: 4,  # Weekend
#     6: 4   # Weekend
# }

# your_dataframe['day_type'] = your_dataframe['day_of_week'].map(day_types)




# df_with_dt = your_dataframe.copy()
# print(df_with_dt)
# your_dataframe = your_dataframe.set_index('datetime')
# df_without_date_name = your_dataframe.drop(['Load','name','level_0_x','precipprob', 'severerisk','uvindex', 'solarenergy','preciptype','snow','day_of_week',], axis=1) 




# df_without_date_name['conditions'] = df_without_date_name['conditions'].astype(str)
# imputer = SimpleImputer(strategy="median")
# df_num = df_without_date_name.drop("conditions", axis=1)
# imputer.fit(df_num)



# zero_columns = df_num.columns[(df_num == 0).all()]
# print("Columns with only zero values:", zero_columns)
# df_num = df_num.drop(zero_columns, axis=1)



# num_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy="median")),
#     ('std_scaler', StandardScaler()),
#     ])
# df_num_tr = num_pipeline.fit_transform(df_num)

# from sklearn.compose import ColumnTransformer
# num_attribs = list(df_num)
# cat_attribs = ["conditions"]
# cat_pipeline = Pipeline([
#     ('onehot', OneHotEncoder())
# ])
# full_pipeline = ColumnTransformer([
# ("num", num_pipeline, num_attribs),
# ("cat", cat_pipeline, cat_attribs),
# ])

# num_prepared = full_pipeline.fit_transform(df_without_date_name)   #df_num_tr 

# # print((num_prepared))
# # df = your_dataframe

# np.savetxt("probaproba.csv", num_prepared, delimiter=',')
# # Data preprocessing and feature engineering (you may need to adjust this based on your data)
# # ...





# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score




# X_train, X_test, y_train, y_test = train_test_split(num_prepared, y, test_size=0.2, random_state=42)
# # # Assuming num_prepared is a DataFrame and y is the corresponding target variable



# # Calculate the number of rows corresponding to the last seven days (assuming hourly data)
# rows_per_day = 24  # Assuming hourly data
# last_seven_days_rows = 7 * rows_per_day

# # Split the data
# # X_train = num_prepared[:-last_seven_days_rows, :]
# # y_train = y[:-last_seven_days_rows]

# X_test = num_prepared[-last_seven_days_rows:, :]
# y_test = y[-last_seven_days_rows:]


# weight_for_second_column = 2.0
# weight_for_first_column = 2.0

# # Create an array of ones with the same shape as X_train and X_test
# feature_weights = np.ones(X_train.shape[1])

# # Apply different weights to specific columns
# feature_weights[1] = weight_for_second_column
# feature_weights[3] = weight_for_first_column

# # Multiply each column by its corresponding weight
# X_train_weighted = X_train * feature_weights
# X_test_weighted = X_test * feature_weights
# # Build FNN model

# model = Sequential([
#     Dense(64, input_dim=X_train.shape[1], activation='relu'),
#     Dense(128, activation='tanh'),
#     Dense(64, activation='LeakyReLU'),
#     Dense(1, activation='linear')
# ])

# model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')



# scaler = MinMaxScaler()

# # Fit the scaler on X_train and transform both X_train and X_test
# X_train_scaled = scaler.fit_transform(X_train_weighted)
# X_test_scaled = scaler.transform(X_test_weighted)



# # model.fit(X_train_weighted, y_train, epochs=100, batch_size=2, verbose=1)

# # y_pred = model.predict(X_test_weighted)




# model.fit(X_train_scaled, y_train, epochs=150, batch_size=2, verbose=1)

# y_pred = model.predict(X_test_scaled)

# mape_final = calculate_mape(y_test, y_pred.flatten())
# print("Final MAPE on Test Set:", mape_final)
















# # import matplotlib.pyplot as plt

# # # Assuming you have the arrays y_test and y_pred_final

# # # Plot real values in blue
# # plt.scatter(range(len(y_test)), y_test, color='blue', label='Real Values')

# # # Plot predicted values in red
# # plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Values')

# # # Add labels and legend
# # plt.xlabel('Index')
# # plt.ylabel('Target Variable')
# # plt.legend()

# # # Show the plot
# # plt.show()













# # print(num_prepared)
# datetime_y = pd.concat([df_with_dt['datetime'], pd.Series(y, name='Load')], axis=1)

# # Split the merged data into training and testing sets for plotting
# # _, datetime_y_test = train_test_split(datetime_y, test_size=0.2, random_state=42)

# datetime_y_test = datetime_y[-last_seven_days_rows:]


# # Assuming you have the arrays y_test and y_pred_final

# # Plot real values in blue
# plt.scatter(datetime_y_test['datetime'], datetime_y_test['Load'], color='blue', label='Real Values')

# # Plot predicted values in red
# plt.scatter(datetime_y_test['datetime'], y_pred.flatten(), color='red', label='Predicted Values')

# # Add labels and legend
# plt.xlabel('Datetime')
# plt.ylabel('Target Variable')
# plt.title('Real vs Predicted Values for the Test Set')
# plt.legend()

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45)

# # Show the plot
# plt.show()












# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader, TensorDataset
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.model_selection import train_test_split
# # import numpy as np


# # X_train, X_test, y_train, y_test = train_test_split(num_prepared, y, test_size=0.2, random_state=42)

# # X_test = X_test[-last_seven_days_rows:, :]
# # y_test = y_test[-last_seven_days_rows:]

# # weight_for_second_column = 2.0
# # weight_for_first_column = 2.0

# # # Apply different weights to specific columns
# # X_train[:, 1] *= weight_for_second_column
# # X_train[:, 3] *= weight_for_first_column
# # X_test[:, 1] *= weight_for_second_column
# # X_test[:, 3] *= weight_for_first_column

# # # Convert to PyTorch tensors
# # X_train_tensor = torch.FloatTensor(X_train)
# # y_train_tensor = torch.FloatTensor(y_train.values)
# # X_test_tensor = torch.FloatTensor(X_test)
# # y_test_tensor = torch.FloatTensor(y_test.values)

# # # Create DataLoader
# # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# # # Build a PyTorch model
# # class Net(nn.Module):
# #     def __init__(self):
# #         super(Net, self).__init__()
# #         self.fc1 = nn.Linear(X_train.shape[1], 64)
# #         self.fc2 = nn.Linear(64, 128)
# #         self.fc3 = nn.Linear(128, 64)
# #         self.fc4 = nn.Linear(64, 1)
# #         self.relu = nn.ReLU()
# #         self.tanh = nn.Tanh()
# #         self.leaky_relu = nn.LeakyReLU()

# #     def forward(self, x):
# #         x = self.relu(self.fc1(x))
# #         x = self.tanh(self.fc2(x))
# #         x = self.leaky_relu(self.fc3(x))
# #         x = self.fc4(x)
# #         return x

# # model = Net()

# # # Define loss function and optimizer
# # criterion = nn.L1Loss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)

# # # Train the model
# # for epoch in range(100):
# #     for inputs, labels in train_loader:
# #         optimizer.zero_grad()
# #         outputs = model(inputs)
# #         loss = criterion(outputs.flatten(), labels)
# #         loss.backward()
# #         optimizer.step()

# # # Evaluate the model on the test set
# # with torch.no_grad():
# #     model.eval()
# #     y_pred = model(X_test_tensor)
# #     mape_final = calculate_mape(y_test_tensor.numpy(), y_pred.flatten().numpy())
# #     print("Final MAPE on Test Set:", mape_final)