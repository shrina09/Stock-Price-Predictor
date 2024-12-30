'''
CIS4780: Computational Intelligence 
Authors: Puneet Sandher, Diya Parmar, Shrina Patel, Adina Mubbashir, Ryan Nguyen
Date: Friday, November 29, 2024
Purpose: This program collects raw stock data 
'''

import time
import yfinance as yf
import os 

# Collects data of a stock and stores it as a csv file
def getStockData(ticker: str, dataFolderPath: str):
    attempts = 0
    dataCollected = 0
    # 3 attempts are made to pull the stock data, incase there are any unexpected issues with the library
    while attempts < 3:
        try:
            timer = time.time()
            # Use yfinance's download method to fetch data
            data = yf.download(ticker, start="1970-01-01", end="2024-06-01")
            # Store data in a csv
            data.to_csv(f"{dataFolderPath}/{ticker}.csv")  
            # Data is pulled successfully
            dataCollected = 1
            return dataCollected
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            attempts += 1
            time.sleep(1)  
    return dataCollected


def main():
    stockList = open('stockList.txt', 'r')
    dataFolderPath = "data"

    # Create folder to store raw data 
    if not os.path.exists(dataFolderPath):
        os.makedirs(dataFolderPath)

    # Iterate through list of stocks and collect data
    for stock in stockList:

        print(f"Processing ticker: {stock}")
        dataCollected = getStockData(stock, dataFolderPath)

        # Display message based on if data was successfully collected
        if dataCollected == 1:
            print(f"SUCCESS: Data for {stock} was downloaded and saved.")
        else:
            print(f"FAIL: Data for {stock} was not downloaded and saved.")

if __name__ == "__main__":
    main()
