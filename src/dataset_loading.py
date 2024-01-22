import pandas as pd
import sys
from util.database import connect_to_database, save_new_data


def process_new_data(filename):
    df = pd.read_csv(filename)
    connection = connect_to_database()
    save_new_data(df, connection)
    connection.close()


if __name__ == "__main__":
    process_new_data(sys.argv[1])
