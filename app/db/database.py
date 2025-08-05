import psycopg2
from psycopg2 import Error

# Define the PostgreSQL database connection parameters
DB_HOST = 'YOUR_DB_HOST'
DB_NAME = 'YOUR_DB_NAME'
DB_USER = 'YOUR_DB_USER'
DB_PASSWORD = 'YOUR_DB_PASSWORD'

# Define a function to connect to the PostgreSQL database
def connect_to_db():
    try:
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST
        )
        return connection
    except Error as e:
        print(f"Error connecting to database: {e}")

# Define a function to create a table in the PostgreSQL database
def create_table():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL
        );
    ''')
    connection.commit()
    cursor.close()
    connection.close()

# Define a function to insert a document into the PostgreSQL database
def insert_document(text):
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute('''
        INSERT INTO documents (text)
        VALUES (%s);
    ''', (text,))
    connection.commit()
    cursor.close()
    connection.close()

# Define a function to retrieve a document from the PostgreSQL database
def retrieve_document(id):
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute('''
        SELECT text
        FROM documents
        WHERE id = %s;
    ''', (id,))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    return result[0] if result else None