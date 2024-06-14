import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def parse_filename(filename):
    """ Parsen des Dateinamens, um Netzwerknamen bis 'ALPHA', Alpha und Noise zu extrahieren """
    pattern = r"ergebnis_(.*?)_ALPHA_(.*?)_NOISE_(.*?).txt"
    match = re.match(pattern, filename)
    if match:
        network_name = match.group(1)  # Extrahiert alles bis 'ALPHA'
        alpha = float(match.group(2))  # Extrahiert den Alpha-Wert
        noise = float(match.group(3))  # Extrahiert den Noise-Wert
        return network_name, alpha, noise
    return None, None, None

def read_data_from_file(file_path):
    """ Einlesen der Daten aus der Datei und Erstellung eines temporären DataFrame """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Episode" in line:
                parts = line.split(',')
                episode = int(parts[0].split(' ')[1].split('/')[0])
                reward = float(parts[1].split(" ")[2])
                cash = float(parts[1].split(" ")[4])
                data.append({'Episode': episode, 'Reward': reward, 'Cash': cash})
    return pd.DataFrame(data)

def load_data(directory):
    """ Laden aller Dateien im Verzeichnis und Zusammenführung in einem DataFrame """
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            network_name, alpha, noise = parse_filename(filename)
            if network_name is not None:
                df = read_data_from_file(os.path.join(directory, filename))
                df['Network'] = network_name
                df['Alpha'] = alpha
                df['Noise'] = noise
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def update_plot(i):
    plt.clf()  # Clear the current figure
    data_frame = load_data('./Ergebnis')  # Load the most recent data
    if not data_frame.empty:
        sns.set_theme(style="whitegrid")
        data_frame = data_frame.groupby(['Network', 'Episode']).mean().reset_index()
        palette = sns.color_palette("hsv", n_colors=data_frame['Network'].nunique())
        for network in data_frame['Network'].unique():
            subset = data_frame[data_frame['Network'] == network]
            alpha = subset['Alpha'].iloc[0]
            noise = subset['Noise'].iloc[0]
            sns.lineplot(x=subset['Episode'], y=subset['Reward'], label=f"{network} α:{alpha:.5f} Noise:{noise:.5f}", linewidth=1.5)
            rolling_data = subset['Reward'].rolling(window=7, center=True).mean()
            sns.lineplot(x=subset['Episode'], y=rolling_data, label=f"{network} (Geglättet) α:{alpha:.5f} Noise:{noise:.5f}", linewidth=2.5, linestyle='--')
        plt.legend(title="Network Info", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.title("Network Performance Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)

def animated_plot():
    fig = plt.figure(figsize=(60, 10))  # Größeres Figure-Objekt für eine bessere Anzeige
    ani = FuncAnimation(fig, update_plot, interval=30000)  # Update every 30 seconds
    plt.show()


animated_plot()
