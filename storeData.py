'''
CIS4780: Computational Intelligence 
Authors: Puneet Sandher, Diya Parmar, Shrina Patel, Adina Mubbashir, Ryan Nguyen
Date: Friday, November 29, 2024
Purpose: This program stores stock data into a SQLite database
'''

import os
import sqlite3
import pandas as pd

# Create a table called stocks in the database, to store raw data 
def setupDatabase(dbPath="testStocks.db"):
    # Connect to sqlite database
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()
    # Create table with the raw data 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            adj_close REAL,
            close REAL,
            high REAL,
            low REAL,
            open REAL,
            volume INTEGER
        )
    ''')
    conn.commit()
    return conn

# Store the data in each file in the table 
def storeInDatabase(filePath, conn):
    # Get the ticker name from the file name
    ticker = os.path.splitext(os.path.basename(filePath))[0]  
    
    # Parse the CSV for each column and store it in the database
    try:
        df = pd.read_csv(filePath, skiprows=3) 
        df.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

        df["Ticker"] = ticker

        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO stocks (ticker, date, adj_close, close, high, low, open, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row["Ticker"], row["Date"], row["Adj Close"], 
                row["Close"], row["High"], row["Low"], row["Open"], row["Volume"]
            ))
        conn.commit()
        print(f'{ticker} stored in database')
    except Exception as e:
        print(f'Error storing data in the database. Message {e}')

# Iterate through each ticker file and store it in the database
def processDirectory(directory, conn):
    for fileName in os.listdir(directory):
        storeInDatabase(os.path.join(directory, fileName), conn)


# Clear the contents of the database
def clearDatabase(conn):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM stocks")  
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='stocks'")
    conn.commit()

def main():
    conn = setupDatabase()
    dataDirectory = "data"
    processDirectory(dataDirectory, conn)
    conn.close()

if __name__ == "__main__":
    main()
