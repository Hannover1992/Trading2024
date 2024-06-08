import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class IDataSource(ABC):
    
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass

class MockSyntheticSinusData(IDataSource):

    def get_data(self) -> pd.DataFrame:
        # Aktuell verwenden wir synthetische Daten
        return self.generate_synthetic_data()

    def generate_synthetic_data(self) -> pd.DataFrame:
        amplitude = 40
        frequenz = 1 / 8
        days = 20
        start_price = 500
        end_price = 500
        noise_level = 10  # Adjust this value to increase or decrease noise

        t = np.arange(0, days)
        linear_trend = np.linspace(start_price, end_price, days)
        sinusoidal = amplitude * np.sin(2 * np.pi * frequenz * t)
        noise = np.random.normal(0, noise_level, days)

        price = linear_trend + sinusoidal + noise
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
