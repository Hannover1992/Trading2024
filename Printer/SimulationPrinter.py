import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class SimulationPrinter:
    @staticmethod
    def plot_results(simulation):
        if simulation.data is None:
            raise ValueError("Data not loaded. Please load data before plotting.")

        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1, 1]})

        # Plot the price data with buy/sell signals
        sns.lineplot(data=simulation.data, x='Tage', y='Preis', ax=ax1)
        buy_signals = [(i, h) for i, h in enumerate(simulation.history) if h['Action'] == 'BUY']
        sell_signals = [(i, h) for i, h in enumerate(simulation.history) if h['Action'] == 'SELL']
        ax1.scatter([i for i, _ in buy_signals], [h['Price'] for _, h in buy_signals], color='green', label='Buy', s=[h['Action Volume'] * 100 for _, h in buy_signals])
        ax1.scatter([i for i, _ in sell_signals], [h['Price'] for _, h in sell_signals], color='red', label='Sell', s=[h['Action Volume'] * 100 for _, h in sell_signals])
        ax1.set_title("Synthetische Preisdaten (Sinus) mit Kauf- und Verkaufsaktionen")
        ax1.set_xlabel("Tage")
        ax1.set_ylabel("Preis")
        ax1.legend()

        # Plot the portfolio value data
        portfolio_values = [{'Cash': h['Balance'], 'Shares': h['Shares'] * h['Price']} for h in simulation.history]
        cash_values = [pv['Cash'] for pv in portfolio_values]
        share_values = [pv['Shares'] for pv in portfolio_values]
        bar_width = 0.4
        r = np.arange(len(simulation.history))

        ax2.bar(r, cash_values, color='blue', edgecolor='grey', width=bar_width, label='Cash')
        ax2.bar(r, share_values, bottom=cash_values, color='orange', edgecolor='grey', width=bar_width, label='Shares')
        ax2.set_title("Portfolio Value (Cash and Shares)")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Value")
        ax2.legend()

        # Plot the reward data
        rewards = [h['Reward'] for h in simulation.history]
        sns.lineplot(x=r, y=rewards, ax=ax3, color='purple')
        ax3.set_title("Rewards Over Time")
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Reward")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_history(simulation):
        history_df = pd.DataFrame(simulation.history)
        print(history_df)
