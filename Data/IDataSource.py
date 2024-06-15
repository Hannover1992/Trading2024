from abc import ABC, abstractmethod
import pandas as pd

class IDataSource(ABC):
    
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass
