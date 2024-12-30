"""
CIS4780: Computational Intelligence 
Authors: Puneet Sandher, Diya Parmar, Shrina Patel, Adina Mubbashir, Ryan Nguyen
Date: Friday, November 29, 2024
Purpose: This file views entries from the stocks_dev database.
"""


import sqlite3

# Connects to the database and prints the first 3000 records in the terminal (for debugging and validation purposes)
def viewEntries(db_path="stocks.db", limit=3000):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM stocks LIMIT {limit}")
    rows = cursor.fetchall()

    for row in rows:
        print(row)

    conn.close()

viewEntries()