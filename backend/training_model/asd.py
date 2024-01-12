from database.database import DatabaseHandler

db_handler = DatabaseHandler()

asd,asd1 = db_handler.read_preprocessed_data()

print(asd['datetime'])