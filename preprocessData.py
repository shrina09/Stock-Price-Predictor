"""
CIS4780: Computational Intelligence 
Authors: Puneet Sandher, Diya Parmar, Shrina Patel, Adina Mubbashir, Ryan Nguyen
Date: Friday, November 29, 2024
Purpose: This file preprocesses, normalizes, and adds features to stock data, storing the results in a database.
"""

import sqlite3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Preprocesses the raw stock data for a given ticker
def preprocessStockData(conn, ticker):
    try:
        # Queries raw data for the ticker
        ticker = ticker + "\n"
        query = f"SELECT * FROM stocks WHERE ticker = '{ticker}'"
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print(f"No data found for ticker {ticker} in stocks table")

        # Converts date column to datetime and drops invalid rows
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date', 'adj_close', 'close', 'high', 'low', 'open', 'volume'], inplace=True)

        # Removes weekends
        df = df[df['date'].dt.weekday < 5]

        df.sort_values(by='date', inplace=True)

        return df
    except Exception as e:
        print(f"Error preprocessing data for ticker {ticker}: {e}")
        return pd.DataFrame()


# Normalizes numerical columns
def normalizeData(df):
    try:
        numericalColumns = ['adj_close', 'close', 'high', 'low', 'open', 'volume']
        scaler = MinMaxScaler()

        # Normalizes every column, except for date and ticker columns
        df[numericalColumns] = scaler.fit_transform(df[numericalColumns])

        return df
    except Exception as e:
        print(f"Error during normalization: {e}")

        return df


# Time-based feature
def addTimeBasedFeature(df):
    try:
        # Adds a 'day_of_week'
        df['day_of_week'] = df['date'].dt.weekday
        return df
    except Exception as e:
        print(f"Error adding time-based feature: {e}")
        return df


# Price difference and daily returns feature
def addPriceDifferenceReturn(df):
    try:
        # Adds 'price_difference'
        df['price_difference'] = df['close'] - df['open']

        # Add 'daily_return'
        df['daily_return'] = df['close'].pct_change().fillna(0)

        return df
    except Exception as e:
        print(f"Error adding new features: {e}")

        return df


# Volume Return (Volume Weighted Average Price) feature
def addVolumeReturn(df):
    try:
        # Calculate Volume Weight
        df['volume_return'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    except Exception as e:
        print(f"Error adding 'volume_return' feature: {e}")
        return df


# Stores preprocessed, normalized, and feature-enhanced data into a new database
def storePreprocessedData(df, conn):
    try:
        # Drops the 'id' column if it exists
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        if 'date' in df.columns:
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # Removes duplicates based on date and ticker
        df = df.drop_duplicates(subset=['date', 'ticker'])

        if not df.empty:
            df.to_sql('stocks_train', conn, if_exists='append', index=False)
        else:
            print(f"WARNING: No valid data to store")
    except Exception as e:
        print(f"Error storing preprocessed data: {e}")


def main():
    try:
        # Connects to original database
        conn = sqlite3.connect("stocks.db")

        # Connects to new database and recreate the table
        newConn = sqlite3.connect("stocks_train.db")

        # Drops the table stock_dev table if it exists to ensure the schema is correct
        newConn.execute("DROP TABLE IF EXISTS stocks_train")
        newConn.execute('''
            CREATE TABLE IF NOT EXISTS stocks_train (
                date DATE NOT NULL,
                adj_close REAL,
                close REAL,
                high REAL,
                low REAL,
                open REAL,
                volume INTEGER,
                ticker TEXT NOT NULL,
                day_of_week INTEGER,
                price_difference REAL,
                daily_return REAL,
                volume_return REAL,
                PRIMARY KEY (date, ticker)
            )
        ''')

        # Reads tickers from stockList.txt
        with open("stockList.txt", 'r') as stock_list:
            tickers = [line.strip() for line in stock_list]

        # Checks if stock list is empty
        if not tickers:
            print("Stock list is empty, exiting the program")
            conn.close()
            newConn.close()
            return

        # Processes each ticker
        for ticker in tickers:
            preprocessedData = preprocessStockData(conn, ticker)
            if not preprocessedData.empty:
                preprocessedData['ticker'] = ticker
                normalizedData = normalizeData(preprocessedData)
                featureEnhancedData = addTimeBasedFeature(normalizedData)
                featureEnhancedData = addPriceDifferenceReturn(featureEnhancedData)
                finalData = addVolumeReturn(featureEnhancedData)
                storePreprocessedData(finalData, newConn)

        # Closes the database connections
        conn.close()
        newConn.close()

        print("Preprocessing, normalization, and features applied to the data and stored")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
