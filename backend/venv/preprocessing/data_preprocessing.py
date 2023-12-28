import pandas as pd
import matplotlib.pyplot as plt
import glob
import os 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sqlalchemy import create_engine
import numpy as np
folder_path_weather = "/home/qnchuck/Desktop/csv_data/NYS Weather Data/New York City, NY"  
file_pattern_weather = "New York City,*.csv"

excel_file_path = '/home/qnchuck/Desktop/csv_data/US Holidays 2018-2021.xlsx'
root_folder_load = "/home/qnchuck/Desktop/csv_data/NYS Load  Data"  


def remove_nulls_from_dataframes(dfs):
    
    dfs = dfs.interpolate(limit_area='inside').fillna(0)
    return dfs

def read_weather_data_from_folder(folder_path, file_pattern):
    files = glob.glob(f"{folder_path}/{file_pattern}")

    #Read files into a pandas DataFrame and remove nulls
    dfs = [pd.read_csv(file) for file in files]
    
    concated_dfs = concat_dataframes_into_df(dfs)
    removed_nulls_dfs = remove_nulls_from_dataframes(concated_dfs)
    
    return removed_nulls_dfs

def read_load_data_from_folder(root_folder):
    all_dfs = []

    for folder_name in os.listdir(root_folder):
        if folder_name.endswith("pal_csv") and os.path.isdir(os.path.join(root_folder, folder_name)):
            folder_path = os.path.join(root_folder, folder_name)

            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(folder_path, filename)
                    # Read CSV into DataFrame
                    df = pd.read_csv(file_path)

                    df = df[df['Name'] == 'N.Y.C.']

                    # Append DataFrame to the list
                    all_dfs.append(df)

    concated_dfs = concat_dataframes_into_df(all_dfs)
    removed_nulls_df = remove_nulls_from_dataframes(concated_dfs)
    return removed_nulls_df

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

    # Reindex the DataFrame with the complete date range to fill missing hours
    df_filled = dfs_to_add_values.reindex(complete_date_range)

    # Reset the index to have 'datetime' as a column again
    df_filled = df_filled.reset_index()
    
    #maybe to move this into merge method, but works well here also
    if date_column=="datetime":
        df_filled['conditions'] = df_filled['conditions'].replace(0, method='ffill') 
        df_filled['conditions'] = df_filled['conditions'].replace(0, method='bfill') # in case there is no previous non zero value

    df_filled = df_filled.interpolate(limit_area='inside').fillna(0)
   
    print(len(df_filled))
    return df_filled
    
def read_and_modify_weather_data():
    dfs_weather = read_weather_data_from_folder(folder_path_weather, file_pattern_weather)
    # dfs_weather = concat_dataframes_into_df(dfs_weather)
    dfs_weather = remove_not_full_hour_values(dfs_weather, 'datetime')
    dfs_weather = add_missing_dates_into_df(dfs_weather,'datetime')
    return dfs_weather

def read_and_modify_load_data():
    dfs_load = read_load_data_from_folder(root_folder_load)
    # dfs_load = concat_dataframes_into_df(dfs_load)
    dfs_load = remove_not_full_hour_values(dfs_load, 'Time Stamp')
    dfs_load = add_missing_dates_into_df(dfs_load, 'Time Stamp')
    return dfs_load

def merge_load_and_weather_data():
    dfs_weather = read_and_modify_weather_data()
    dfs_load = read_and_modify_load_data()

    df_weather_reset = dfs_weather.reset_index()
    # Rename the "index" column to "datetime"
    df_weather_reset = df_weather_reset.rename(columns={'index': 'datetime'})
    
    
    df_load_reset = dfs_load.reset_index()
    # Rename the "index" column to "datetime"
    df_load_reset = df_load_reset.rename(columns={'index': 'Time Stamp'})


    df_merged = pd.merge(df_weather_reset, df_load_reset, left_on='datetime', right_on='Time Stamp', how='inner')

    # Drop redundant datetime column from df_load
    df_merged = df_merged.drop(['Time Stamp', 'Time Zone', 'Name', 'PTID','level_0_y'], axis=1)

    df_merged.to_csv('merged_data.csv', index=False)
    return df_merged

def remove_special_dates(df):
    # Read the Excel file into a DataFrame
    df_special = pd.read_excel(excel_file_path)
    
    special_dates = df_special['Unnamed: 2']

    # Loop through each DataFrame in the list
   
    df = df[~df['datetime'].dt.date.isin(special_dates.dt.date)]
     
    print(len(df))
    return df

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



def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# Example usage:
# Replace 'your_dataframe', 'your_table_name', and 'your_connection_string' with your actual DataFrame, table name, and connection string.


your_dataframe = merge_load_and_weather_data()

cat_encoder = OneHotEncoder()
df_cat = your_dataframe[["conditions"]]
df_cat['conditions'] = df_cat['conditions'].replace(0, method='ffill') 
df_cat['conditions'] = df_cat['conditions'].replace(0, method='bfill') # in case there is no previous non zero value


print(df_cat)
df_cat_1hot = cat_encoder.fit_transform(df_cat)
your_dataframe = remove_special_dates(your_dataframe)
y = your_dataframe['Load'] 
your_dataframe = your_dataframe.sort_values(by='datetime')



# Calculate the Z-scores for each value in the column
z_scores = (your_dataframe['temp'] - your_dataframe['temp'].mean()) / your_dataframe['temp'].std()

# Define a threshold for Z-scores (e.g., 3 for extreme outliers)
threshold = 3

# Create a boolean mask to identify extreme values
outliers_mask = (abs(z_scores) > threshold)

# Remove extreme outliers and create a new DataFrame
df_no_outliers = your_dataframe[~outliers_mask] 

# Interpolate missing values in the 'temp' column
df_no_outliers['temp'] = df_no_outliers['temp'].interpolate(limit_area='inside').fillna(0)


your_dataframe['temp'] = df_no_outliers['temp'].copy()


#Calculating of the temperature one day before
your_dataframe['previous_day_temp'] = your_dataframe.groupby(your_dataframe['datetime'].dt.hour)['temp'].shift(24)
your_dataframe['temp_seven_days_before'] = your_dataframe['temp'].shift(7 * 24)  # Assuming your data is hourly
your_dataframe['datetime'] = pd.to_datetime(your_dataframe['datetime'])




your_dataframe = your_dataframe.set_index('datetime')

df_without_date_name = your_dataframe.drop(['Load','name','level_0_x','precipprob', 'severerisk','uvindex', 'solarenergy','preciptype','snow',], axis=1) 

df_without_date_name['conditions'] = df_without_date_name['conditions'].astype(str)
imputer = SimpleImputer(strategy="median")
df_num = df_without_date_name.drop("conditions", axis=1)
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

num_prepared = full_pipeline.fit_transform(df_without_date_name)

print((num_prepared))
# df = your_dataframe

np.savetxt("probaproba.csv", num_prepared, delimiter=',')
# Data preprocessing and feature engineering (you may need to adjust this based on your data)
# ...

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# Split the data
# X = data.drop(['Load', 'datetime','conditions','name','level_0_x'], axis=1)  # Features
# X.to_csv('merged_data1.csv', index=False)
# y = data['Load']  # Target variable





# Create and train the model (using RandomForestRegressor instead of LinearRegression)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Plot predictions vs real data with limited axis range
# plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Actual vs Predicted')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
# plt.xlabel("Actual Load")
# plt.ylabel("Predicted Load")
# plt.title("Predictions vs Real Data")
# plt.legend()
# plt.xlim(min(y_test), max(y_test))
# plt.ylim(min(y_test), max(y_test))
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(num_prepared, y, test_size=0.2, random_state=42)

weight_for_second_column = 2.0
weight_for_fourth_column = 2.0

# Create an array of ones with the same shape as X_train and X_test
feature_weights = np.ones(X_train.shape[1])

# Apply different weights to specific columns
feature_weights[1] = weight_for_second_column
feature_weights[3] = weight_for_fourth_column

# Multiply each column by its corresponding weight
X_train_weighted = X_train * feature_weights
X_test_weighted = X_test * feature_weights
# Build FNN model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model with verbose output and calculate MAPE during training
epochs = 100
for epoch in range(epochs):
    X_train_weighted = X_train * feature_weights
    # Fit the model
    model.fit(X_train_weighted, y_train, epochs=1, batch_size=2, verbose=1)
    
    # Apply custom weights to input features for validation set
    X_test_weighted = X_test * feature_weights
    # Make predictions on the validation set
    y_pred = model.predict(X_test_weighted)
    
    # Calculate and print MAPE for the current epoch
    mape = calculate_mape(y_test, y_pred.flatten())
   
   
   #previous version
    # model.fit(X_train, y_train, epochs=1, batch_size=2, verbose=1)
    
    # # Make predictions on the validation set
    # y_pred = model.predict(X_test)
    
    # # Calculate and print MAPE for the current epoch
    # mape = calculate_mape(y_test, y_pred.flatten())
    #previous version
    
    print(f"Epoch {epoch + 1}/{epochs} - MAPE: {mape:.2f}%")

# Final evaluation
y_pred_final = model.predict(X_test_weighted)
mape_final = calculate_mape(y_test, y_pred_final.flatten())
print("Final MAPE on Test Set:", mape_final)


# import xgboost as xgb
# dtrain = xgb.DMatrix(X, label=y)

# # Set parameters
# params = {"objective": "reg:squarederror", "max_depth": 3}

# # Train the model
# xg_model = xgb.train(params, dtrain)

# # Make predictions
# y_pred = xg_model.predict(dtrain)

# # Calculate MAPE
# mape = np.mean(np.abs((y - y_pred) / y)) * 100
# print("XGBoost MAPE:", mape)


# import lightgbm as lgb

# # Convert data to Dataset format
# lgb_train = lgb.Dataset(X, label=y)

# # Set parameters
# params = {"objective": "regression", "max_depth": 3}

# # Train the model
# lgb_model = lgb.train(params, lgb_train)

# # Make predictions
# y_pred = lgb_model.predict(X)

# # Calculate MAPE
# mape = np.mean(np.abs((y - y_pred) / y)) * 100
# print("LightGBM MAPE:", mape)





# def calculate_mape(actual, predicted):
#     """
#     Calculate Mean Absolute Percentage Error (MAPE).

#     Parameters:
#     actual (list or array): Actual values.
#     predicted (list or array): Predicted values.

#     Returns:
#     float: MAPE value.
#     """
#     actual, predicted = np.array(actual), np.array(predicted)
#     return np.mean(np.abs((actual - predicted) / actual)) * 100

# y_true = y_test
# # Example usage:
# mape_value = calculate_mape(y_test, y_pred)
# print(f"MAPE: {mape_value:.2f}%")

# mae = mean_absolute_error(y_true, y_pred)
# print(f"Mean Absolute Error (MAE): {mae}")

# # Mean Squared Error (MSE)
# mse = mean_squared_error(y_true, y_pred)
# print(f"Mean Squared Error (MSE): {mse}")

# # Root Mean Squared Error (RMSE)
# rmse = np.sqrt(mse)
# print(f"Root Mean Squared Error (RMSE): {rmse}")

# # Mean Absolute Percentage Error (MAPE)
# mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# # R-squared (R2 Score)
# r2 = r2_score(y_true, y_pred)
# print(f"R-squared (R2 Score): {r2:.4f}")
# # write_df_to_postgresql(your_dataframe, 'forecast', 'postgresql://postgres:admin@localhost:5432/forecast')


# def evaluate_model_performance(mape, rmse, r2):
#     """
#     Categorize model performance based on error metrics.

#     Parameters:
#     mape (float): Mean Absolute Percentage Error.
#     rmse (float): Root Mean Squared Error.
#     r2 (float): R-squared.

#     Returns:
#     str: Model performance category.
#     """
#     if mape < 5 and rmse < 5 and r2 > 0.9:
#         return "Excellent"
#     elif mape < 10 and rmse < 10 and r2 > 0.8:
#         return "Good"
#     elif mape < 20 and rmse < 20 and r2 > 0.7:
#         return "Moderate"
#     else:
#         return "Poor"


# model_category = evaluate_model_performance(mape, rmse, r2)
# print(f"Model Performance: {model_category}")
# #df = pd.concat(all_dataframes, ignore_index=True)

# # #df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])

# # df = df[df["Time Stamp"].dt.minute == 0]

# # df = df.sort_values(by='Time Stamp')
# # df = df.interpolate(limit_area='inside').fillna(0)
# # Filter data for May 2019
# # start_date = '2020-10-20'
# # end_date = '2020-11-20'
# # filtered_data = df[(df['Time Stamp'] >= start_date) & (df['Time Stamp'] <= end_date)]

# # # Plot the data
# # plt.figure(figsize=(12, 6))

# # # Example: Plot load for each 'Name'
# # for name, group in filtered_data.groupby('Name'):
# #     plt.plot(group['Time Stamp'], group['Load'], label=name)
# # print(filtered_data)
# # # Save the filtered_data DataFrame to a new CSV file
# # filtered_data.to_csv('filtered_data.csv', index=False)

# # plt.title('Load for May 2019')
# # plt.xlabel('Time Stamp')
# # plt.ylabel('Load')
# # plt.legend()
# # plt.show()

# def compare_rmse_to_range(actual, predicted):
#     # Ensure actual and predicted are numpy arrays for ease of calculation
#     actual = np.array(actual)
#     predicted = np.array(predicted)

#     # Calculate RMSE
#     rmse = np.sqrt(np.mean((actual - predicted) ** 2))

#     # Calculate the range of the data
#     data_range = np.max(actual) - np.min(actual)

#     # Compare RMSE to the range
#     rmse_to_range_ratio = rmse / data_range

#     return rmse, data_range, rmse_to_range_ratio


# rmse_value, data_range, rmse_to_range_ratio = compare_rmse_to_range(y_true, y_pred)

# print("RMSE:", rmse_value)
# print("Data Range:", data_range)
# print("RMSE to Range Ratio:", rmse_to_range_ratio)











#combined_df = pd.concat(all_dataframes, ignore_index=True)

#combined_df['Time Stamp'] = pd.to_datetime(combined_df['Time Stamp'])
# Order the DataFrame by 'Time Stamp'
#combined_df = combined_df.sort_values(by='Time Stamp')

#combined_df = combined_df[combined_df["Time Stamp"].dt.minute == 0]

#combined_df = combined_df.interpolate(limit_area='inside').fillna(0)
#combined_df.set_index('Time Stamp', inplace = True)

#print(combined_df.head())
#print(combined_df)

# Plot the DataFrame
#plt.plot(df['first'])
#plt.show()
