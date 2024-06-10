import pandas as pd

# Load the data from bitcoin.csv
df = pd.read_csv('./Data/bitcoin.csv')

# Display the first few rows of the DataFrame to check its contents
print(df.head())
