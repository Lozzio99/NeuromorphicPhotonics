import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from classes.neuron import LIFNeuron
from utils import runner
from utils.config import RESULTS_DIRECTORY

length_range = np.arange(50, 301, 10)
strength_range = np.arange(-0.4, 0.401, 0.05)

num_runs = 100
t1 = 100

NEW_RUN = True
filepath = os.path.join(RESULTS_DIRECTORY, f'spike_probability_by_pulse_{num_runs}_runs_2.csv')


def run_new_data(lr, sr):
    results = {
        'strength': [],
        'length': [],
        'spike_probability': []
    }

    loaded['strength'] = loaded['strength'].map(lambda x: round(x, 3))

    for length in lr:
        for strength in sr:
            strength = round(strength, 3)
            loaded_ = loaded.loc[(loaded['strength'] == strength) & (loaded['length'] == length)]
            if len(loaded_) > 0:
                # print(f'looking for strength {strength} and length {length}')
                spike_probability = loaded_['spike_probability'].values[0]
            else:
                if (abs(strength) == 0) | (abs(length) == 0):
                    # print('skipping 0 pulse')

                    spike_probability = 0
                else:
                    pos_pulse = lambda t: strength if t1 < t < t1+length else 0

                    single_neuron = LIFNeuron(pulse_f=pos_pulse)
                    runs = runner.multiple_runs_simulation(single_neuron, n_runs=num_runs, filter_vars=['binary_state'])

                    spikes = [1 if sum(runs[i]['binary_state']) > 0 else 0 for i in runs]
                    spike_probability = sum(spikes) / len(spikes)

                results['strength'].append(strength)
                results['length'].append(length)
                results['spike_probability'].append(spike_probability)

            print(length, f'{strength:.3f}', '\t:', spike_probability)
    return results

def save(results):
    if isinstance(results, dict):
        saved = pd.DataFrame(results)
    elif isinstance(results, pd.DataFrame):
        saved = results
    else:
        raise Exception('Unknown data type to be saved')

    def map0(x):
        xv = eval(x)
        if abs(xv) == 0:
            xv = 0
        return xv

    print('Here')
    saved["strength"] = saved["strength"].map("{:.3f}".format).map(map0)
    saved.to_csv(filepath, index=False)
    return saved

def combine(odf, ndf):
    merged = pd.concat([odf, ndf], ignore_index=True)
    merged = merged.sort_values(by=['length', 'strength']).reset_index(drop=True)
    return merged

def load_results(fp=filepath):
    return pd.read_csv(fp)

def plot(df):
    pivot_df = df.pivot(index='strength', columns='length', values='spike_probability')

    Y, X = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        vmin=0, vmax=1,  # force colorbar limits between 0 and 1
        linewidth=0, antialiased=True
    )

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Spike Probability (0-1)')

    # Labels
    ax.set_ylabel('Pulse Length')
    ax.set_xlabel('Pulse Strength')
    ax.set_zlabel('Spike Probability')

    ax.view_init(elev=30, azim=260)

    plt.tight_layout()
    plt.show()


loaded = load_results()

if NEW_RUN:
    new_data = run_new_data(length_range, strength_range)
    new_df = pd.DataFrame(new_data)
    combined = combine(loaded, new_df)
    save(combined)
    plot(load_results())
else:
    plot(loaded)

