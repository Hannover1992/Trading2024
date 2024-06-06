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
        frequenz = 1 / 7
        days = 8
        offset = 16

        t = np.arange(0, days)
        price = amplitude * np.sin(2 * np.pi * frequenz * t) + offset
        # self.show_plot(t,price)
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
