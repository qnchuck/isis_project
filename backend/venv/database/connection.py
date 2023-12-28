import psycopg2
from psycopg2 import sql

# Replace these with your actual database connection details
dbname = "forecast"
user = "postgres"
password = "admin"
host = "localhost"
port = "5432"

# Establish a connection to the PostgreSQL database
try:
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    print("Connected to the database.")

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Example: Execute a simple query
    query = sql.SQL("SELECT * FROM proba;")
    cursor.execute(query)

    # Fetch all rows
    rows = cursor.fetchall()
    
    # Print the fetched rows
    for row in rows:
        print(row)

except psycopg2.Error as e:
    print("Unable to connect to the database.")
    print(e)

finally:
    # Close the cursor and connection
    if conn:
        conn.close()
        print("Connection closed.")
