import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class IDataSource(ABC):
    
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass

class BitcoinData(IDataSource):
    def __init__(self, file_path):
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        # Load the Bitcoin data from the specified CSV file
        df = pd.read_csv(self.file_path)
        
        # Assuming '24h High (USD)' is treated as the closing price
        # Change '24h High (USD)' to your actual closing price column if different
        t = df.index
        price = df['24h High (USD)'].values
        # df = df[['Date', '24h High (USD)']]  
        # df.columns = ['Days', 'Price']  # Renaming columns to 'Days' and 'Price'
        # self.show_plot(t, price)
        return pd.DataFrame({'Tage': t, 'Preis': price})

    def show_plot(self, x, y):
        sns.set(style="whitegrid")
        sns.lineplot(x=x, y=y)
        plt.title("Synthetische Preisdaten (Sinus)")
        plt.xlabel("Tage")
        plt.ylabel("Preis")
        plt.show()

class MockSyntheticSinusData(IDataSource):

    def get_data(self) -> pd.DataFrame:
        # Aktuell verwenden wir synthetische Daten
        return self.generate_synthetic_data()

    def generate_synthetic_data(self) -> pd.DataFrame:
        days = 16
        start_price = 500
        end_price = 500
        noise_level = 50  # Significantly increased noise

        t = np.arange(0, days)
        linear_trend = np.linspace(start_price, end_price, days)
        c = 1000

        # Multiple sinusoidal patterns with varying parameters
        sinusoidal = sum(
            np.random.randint(100, 300) * np.sin(2 * np.pi * np.random.uniform(1/50, 1/20) * t)
            for _ in range(10)
        )

        noise = np.random.normal(0, noise_level, days)
        #assert that the price is always positive

        price = linear_trend + sinusoidal + noise + c
        price = np.maximum(price, 50)
        # self.show_plot(t, price)
        return pd.DataFrame({'Tage': t, 'Preis': price})

    def plot_synthetic_data(self, data: pd.DataFrame):
        sns.set(style="whitegrid")
        sns.lineplot(data=data, x='Tage', y='Preis')
        plt.title("Synthetische Preisdaten (Sinus)")
        plt.xlabel("Tage")
        plt.ylabel("Preis")
        plt.show()

    def show_plot(self, x, y):
        sns.set(style="whitegrid")
        sns.lineplot(x=x, y=y)
        plt.title("Synthetische Preisdaten (Sinus)")
        plt.xlabel("Tage")
        plt.ylabel("Preis")
        plt.show()
