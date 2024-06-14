import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt


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

# Rest des Skripts bleibt unverändert

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
    return pd.concat(all_data, ignore_index=True)


def plot_network_grauer_interwall(data_frame):
    """
    Erstellt einen Seaborn-Plot für Netzwerk-Performance-Daten.

    :param data_frame: Pandas DataFrame mit den Spalten 'Episode', 'Reward', 'Network', 'Alpha', 'Noise'
    """
    sns.set_theme(style="ticks")

    # Erstellen des Plots
    g = sns.relplot(
        data=data_frame,
        x="Episode", y="Reward",
        hue="Network", style="Network",  # Kategorische Unterscheidung nach Netzwerk
        kind="line",
        height=5, aspect=1.5,  # Größe und Seitenverhältnis des Plots anpassen
        facet_kws={'sharex': False, 'sharey': False}
    )

    # Legende anpassen
    handles, labels = g.ax.get_legend_handles_labels()
    new_labels = []
    for label in labels[1:]:  # Die ersten Labels überspringen, die der Titel sind
        alpha = data_frame[data_frame['Network'] == label]['Alpha'].unique()[0]
        noise = data_frame[data_frame['Network'] == label]['Noise'].unique()[0]
        new_label = f"{label} α:{alpha} Noise:{noise}"
        new_labels.append(new_label)
    g.ax.legend(handles=handles[1:], labels=new_labels, title="Network Info")  # Neue Legende setzen

    plt.show()


def plot_network_performance(data_frame):
    """
    Erstellt einen Seaborn-Plot für Netzwerk-Performance-Daten mit individuellen und geglätteten Trendlinien.

    :param data_frame: Pandas DataFrame mit den Spalten 'Episode', 'Reward', 'Network', 'Alpha', 'Noise'
    """
    sns.set_theme(style="whitegrid")

    # Bereite Daten für das Plotten vor, indem das Figure-Objekt mit einer angepassten Größe erstellt wird
    plt.figure(figsize=(60, 10))  # Größeres Figure-Objekt für eine bessere Anzeige

    # Gruppieren der Daten und Berechnen des Mittelwerts für jede Gruppe
    data_frame = data_frame.groupby(['Network', 'Episode']).mean().reset_index()

    # Palette für die Netzwerke generieren
    palette = sns.color_palette("hsv", n_colors=data_frame['Network'].nunique())

    # Iteriere über jede Netzwerkgruppe für das Plotten
    for network in data_frame['Network'].unique():
        subset = data_frame[data_frame['Network'] == network]
        alpha = subset['Alpha'].iloc[0]
        noise = subset['Noise'].iloc[0]

        # Individuelle Datenlinien zeichnen
        sns.lineplot(x=subset['Episode'], y=subset['Reward'], label=f"{network} α:{alpha:.5f} Noise:{noise:.5f}",
                     linewidth=1.5, color=palette.pop(0))

        # Geglättete Datenlinien zeichnen
        rolling_data = subset['Reward'].rolling(window=7, center=True).mean()
        sns.lineplot(x=subset['Episode'], y=rolling_data, label=f"{network} (Geglättet) α:{alpha:.5f} Noise:{noise:.5f}",
                     linewidth=2.5, linestyle='--')

    # Legende außerhalb des Plots positionieren, rechte Seite
    plt.legend(title="Network Info", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.title("Network Performance Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    # Layout anpassen, um Überlappungen zu vermeiden
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Justiert den rechten Rand des Plots

    plt.show()


# Load your data (replace with your actual loading logic)
data_frame = load_data('./Ergebnis')

# Call the plotting function
# plot_network_grauer_interwall(data_frame)
plot_network_performance(data_frame)

