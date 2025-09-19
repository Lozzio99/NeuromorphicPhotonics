import os

import numpy as np
import pandas as pd

from classes.neuron import LIFNeuron
from utils import runner
from utils.config import RESULTS_DIRECTORY
from utils.plots import plot_pulse_shape_spike_probability

t1 = 100
tx = 0

def pulse_f(length, strength):
    t2 = t1 + length
    t2x = t2 + tx
    t3 = t2x + length

    return lambda t: strength if t1<=t<t2 else -strength if t2x<=t<t3 else 0

NEW_RUN = True
length_range = np.arange(20, 301, 5)
strength_range = np.arange(-0.5, 0.501, 0.025)

num_runs = 100
filepath = os.path.join(RESULTS_DIRECTORY, f'double_spike_{num_runs}_runs.csv')


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
                    single_neuron = LIFNeuron(pulse_f=pulse_f(length, strength))
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

    saved["strength"] = saved["strength"].map("{:.3f}".format).map(map0)
    saved.to_csv(filepath, index=False)
    return saved

def combine(odf, ndf):
    merged = pd.concat([odf, ndf], ignore_index=True)
    merged = merged.sort_values(by=['length', 'strength']).reset_index(drop=True)
    return merged

def load_results(fp=filepath):
    return pd.read_csv(fp)


try:
    loaded = load_results()
except FileNotFoundError as e:
    loaded = None


if NEW_RUN:
    new_data = run_new_data(length_range, strength_range)
    new_df = pd.DataFrame(new_data)
    combined = combine(loaded, new_df)
    save(combined)
    plot_pulse_shape_spike_probability(load_results())
else:
    plot_pulse_shape_spike_probability(loaded)

