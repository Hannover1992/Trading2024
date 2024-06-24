import pandas as pd
from Data.IDataSource import IDataSource
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Config import DATA_LENGTH

class BitcoinData(IDataSource):
    def __init__(self, file_path):
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        # Load the Bitcoin data from the specified CSV file
        df = pd.read_csv(self.file_path)
        
        # Assuming '24h High (USD)' is treated as the closing price
        # Change '24h High (USD)' to your actual closing price column if different
        t = df.index
        #t from 1500 to 2500

        start_index = 1700
        end_index = start_index + DATA_LENGTH

        t = np.arange(start_index, end_index)
        price = df['24h High (USD)'].values
        #price from 1500 to 2500
        price = price[start_index:end_index]

        filtered_df = df.loc[start_index:end_index-1, '24h High (USD)'].reset_index(drop=True)

        result_df = pd.DataFrame({
                    'Days': np.arange(0 , len(filtered_df)),
                    'Preis': filtered_df.values
                })
        # df = df[['Date', '24h High (USD)']]  
        # df.columns = ['Days', 'Price']  # Renaming columns to 'Days' and 'Price'
        # self.show_plot(result_df['Days'], result_df['Preis'])
        return result_df

    def show_plot(self, x, y):
        sns.set(style="whitegrid")
        sns.lineplot(x=x, y=y)
        plt.title("Synthetische Preisdaten (Sinus)")
        plt.xlabel("Tage")
        plt.ylabel("Preis")
        plt.show()
