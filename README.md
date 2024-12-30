# Stock Price Predictor

```
cd existing_repo
git remote add origin https://gitlab.socs.uoguelph.ca/cis4780-stock-price-predictor/stock-price-predictor.git
git branch -M main
git push -uf origin main
```

## Overview
This repository contains a machine learning model designed to predict when to buy and sell stocks. It uses historical stock price data and implements a Long Short-Term Memory (LSTM) neural network for model training. 
Key components of this repo:

- **Data Collection:** Gathering and preparing historical stock data for analysis.
- **Data Preprocessing:** Cleaning, transforming, and feature engineering the data to ensure compatibility with the LSTM model.
- **Model Training:** Using the LSTM to predict optimal buy and sell.

This project aims to assist in making informed trading decisions based on data-driven insights.

## Running Data Collector 

1. Create virutal environment (on Mac)
`python3 -m venv venv`
`source venv/bin/activate`

2. On Visual Studio Code make sure you're using the python interpreter with your venv, check your settings (shift + command +p )

3. Install dependencies 

`pip3 install -r requirements.txt`
Note: Use a Python version higher than 7.8

4. Run collector 
`python3 collector.py`
This will take approximately 35 minutes to fetch all the data 

5. Store data in database 
`python3 storeData.py`
If you need to rerun this script or need to clear the database, the function is in the script, so edit that first to remove everything first. 

You can query data in the database by doing `python3 queryDatabase.py`

## Running the Preprocessor 
1. Complete data collecting
2. Run the preprocessor which preprocesses, normalizes and feature enchances the raw data
`python3 preprocessData.py`

## Training the Model
1. Complete data preprocessing
2. Run the model trainer
`python3 modelTraining.py`

**Note:** This step will take 12 hours minimum, use python 3.8, older versions won't work with tensorflow

## Author Information
Your name: Shrina Patel<br />
Email: shrinapatel359@gmail.com

