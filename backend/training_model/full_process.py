from data_preprocessing_copy import *
from model_creation import ModelCreation
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from db.database import DatabaseHandler

db_handler = DatabaseHandler()

model_create = ModelCreation()
num_prepared, y, df_with_dates = do_final_preparations_for_model('2018-01-01','2021-10-01')

# print(y)
df_with_dates, num_prepared = db_handler.read_preprocessed_data()
# print(df_with_dates)
# print(num_prepared.head())


num_prepared = model_create.fit_transform_pipeline(num_prepared)

X_train, X_test, y_train, y_test = train_test_split(num_prepared, y, test_size=0.2, random_state=42)

        # X_train, X_test = model_create.set_model_weights(X_train, X_test)

rows_per_day = 24  # Assuming hourly data
last_seven_days_rows = 7 * rows_per_day


X_test = num_prepared[-last_seven_days_rows:, :]
y_test = y[-last_seven_days_rows:]

model = model_create.create_model(X_train)

train_pred, test_pred = model_create.compile_fit_predict(X_train, X_test, y_train)

# loaded_model = load_model('my_model.h5')

# Assuming you have new data for forecasting, replace X_new with your actual data
# Make predictions using the loaded model
# predictions = loaded_model.predict(X_test)
# predictions = model.predict(X_test)

# Print or use the predictions as needed
# print(predictions)
    
# mape_final = calculate_mape(y_test, predictions.flatten())
mape_final = calculate_mape(y_test, test_pred.flatten())
print("Final MAPE on Test Set:", mape_final)




# print(num_prepared)
# print(df_with_dates['datetime'])
datetime_y = pd.concat([df_with_dates['datetime'], pd.Series(y, name='Load')], axis=1)

# Split the merged data into training and testing sets for plotting
# _, datetime_y_test = train_test_split(datetime_y, test_size=0.2, random_state=42)

datetime_y_test = datetime_y[-last_seven_days_rows:]


# Assuming you have the arrays y_test and y_pred_final

# Plot real values in blue
plt.scatter(datetime_y_test['datetime'], datetime_y_test['Load'], color='blue', label='Real Values')

# Plot predicted values in red
# plt.scatter(datetime_y_test['datetime'], predictions.flatten(), color='red', label='Predicted Values')
plt.scatter(datetime_y_test['datetime'], test_pred.flatten(), color='red', label='Predicted Values')

# Add labels and legend
plt.xlabel('Datetime')
plt.ylabel('Target Variable')
plt.title('Real vs Predicted Values for the Test Set')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


