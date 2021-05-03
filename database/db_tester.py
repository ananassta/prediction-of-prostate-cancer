import numpy as np
import sqlite3 as sl
import pandas as pd

# Create a connection to our database
# In our example, it's just a base file
conn = sl.connect('prostate_cancer.db')

# Create a cursor
cursor = conn.cursor()

# Make a SELECT query to the database using SQL syntax
cursor.execute("SELECT * FROM PATIENT")

# Get the result of the request
results = cursor.fetchall()

# Show them
print(results)

# Close the database connection
conn.close()
