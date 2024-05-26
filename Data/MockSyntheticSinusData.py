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
        amplitude = 500
        frequenz = 0.01
        days = 365
        offset = 501

        t = np.arange(0, days)
        price = amplitude * np.sin(2 * np.pi * frequenz * t) + offset
        return pd.DataFrame({'Tage': t, 'Preis': price})

    def plot_synthetic_data(self, data: pd.DataFrame):
        sns.set(style="whitegrid")
        sns.lineplot(data=data, x='Tage', y='Preis')
        plt.title("Synthetische Preisdaten (Sinus)")
        plt.xlabel("Tage")
        plt.ylabel("Preis")
        plt.show()
