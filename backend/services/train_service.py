# from model_training_service import ModelTrainingService
from training_model.model_creation import ModelCreation
from training_model.data_preprocessing_copy import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


class TrainService:
    def __init__(self):
        self.model_creation = ModelCreation()

    def start_training(self, date_from, date_to):
        try:
            model_create = ModelCreation()
        
            y = db_handler.read_load_data_by_date(date_from, date_to)
            print(y)
            df_with_dates, df_without_dates = db_handler.read_preprocessed_data(date_from, date_to)
            print(df_with_dates.shape)
            print(df_without_dates.shape)
            
            y = y['Load']
            num_prepared = model_create.fit_transform_pipeline(df_without_dates)
            
            X_train, X_test, y_train, y_test = train_test_split(num_prepared, y, test_size=0.2, random_state=42)
            
            
            rows_per_day = 24  # Assuming hourly data
            last_seven_days_rows = 7 * rows_per_day


            # X_test = num_prepared[-last_seven_days_rows:, :]
            # y_test = y[-last_seven_days_rows:]

            model = model_create.create_model(X_train)

            train_pred, test_pred = model_create.compile_fit_predict(X_train, X_test, y_train)

            path = '/home/qnchuck/Desktop/isis/backend/models/'
            loaded_model = load_model('my_model_2.h5')

            # Assuming you have new data for forecasting, replace X_new with your actual data
            # Make predictions using the loaded model
            # predictions = loaded_model.predict(X_test)
            # predictions = model.predict(X_test)

            # Print or use the predictions as needed
            # print(predictions)
                
            # mape_final = calculate_mape(y_test, predictions.flatten())
            mape_final = calculate_mape(y_test, test_pred.flatten())
            print("Final MAPE on Test Set:", mape_final)
            # mape_final = calculate_mape(y_test, predictions.flatten())
            # print("Final MAPE on Test MODEL 2 Set:", mape_final)
            mape_final = calculate_mape(y_train, train_pred.flatten())
            print("Final MAPE on Train Set:", mape_final)


            mape_str = str(mape_final).replace('.', '_')

            # Construct the filename using the MAPE value
            file_name = f'model_{mape_str}.h5'
            file_path = path+file_name
            model_create.model.save( file_path + '.h5')



            # print(num_prepared)
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

            # Call the data layer (model training service) to start the training
            return True
        except Exception as e:
            print(e)
            return False

    def preprocess_data(self, date_from, date_to):
        try:
            do_final_preparations_for_model(date_from, date_to)
            return True
        except:
            return False