import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Data.IDataSource import IDataSource
from Config import SYNTHETIC_DATA_LENGTH

class MockSyntheticSinusData(IDataSource):

    def get_data(self) -> pd.DataFrame:
        # Aktuell verwenden wir synthetische Daten
        return self.generate_synthetic_data()

    def generate_synthetic_data(self) -> pd.DataFrame:

        np.random.seed(123)
        days = SYNTHETIC_DATA_LENGTH
        start_price = 240
        end_price = 120
        noise_level = 50  # Significantly increased noise

        t = np.arange(0, days)
        linear_trend = np.linspace(start_price, end_price, days)
        c = 150

        # Multiple sinusoidal patterns with varying parameters
        sinusoidal = sum(
            np.random.randint(50, 100) * np.sin(2 * np.pi * np.random.uniform(1/50, 1/3) * t)
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
