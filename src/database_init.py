from util.database import connect_to_database, init_db_schema

if __name__ == "__main__":
    connection = connect_to_database()
    init_db_schema(connection)
    connection.close()
