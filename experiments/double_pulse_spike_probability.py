import os
import threading

import numpy as np
import pandas as pd
import tkinter as tk

from classes.neuron import LIFNeuron
from utils import runner
from utils.config import RESULTS_DIRECTORY
from utils.plots import plot_pulse_shape_spike_probability

t1 = 1100
tx = 0

def pulse_f(length, strength):
    t2 = t1 + length
    t2x = t2 + tx
    t3 = t2x + length

    return lambda t: strength if t1<=t<t2 else -strength if t2x<=t<t3 else 0

length_range = np.arange(50, 301, 5)
strength_range = np.arange(-0.5, 0.501, 0.1)

num_runs = 10
filepath = os.path.join(RESULTS_DIRECTORY, f'double_spike_{num_runs}_runs.csv')


def run_remaining():
    global results, length_range, strength_range

    for length in length_range:
        for strength in strength_range:
            loaded_ = results.loc[(results['strength'] == round(strength, 3)) & (results['length'] == length)]

            if len(loaded_) > 0:
                spike_probability = loaded_['spike_probability'].values[0]
            else:
                single_neuron = LIFNeuron(pulse_f=pulse_f(length, strength))
                runs = runner.multiple_runs_simulation(single_neuron, n_runs=num_runs, filter_vars=['binary_state'])

                spikes = [1 if sum(runs[i]['binary_state']) > 0 else 0 for i in runs]
                spike_probability = sum(spikes) / len(spikes)

                results.loc[len(results)] = {
                    'length': length,
                    'strength': strength,
                    'spike_probability': spike_probability
                }

            print(length, f'{strength:.3f}', '\t:', spike_probability)
    return results

def gui_loop():
    root = tk.Tk()
    root.title("Save Results")
    root.geometry("200x100")

    save_btn = tk.Button(root, text="Save", command=save_results)
    save_btn.pack(expand=True)

    root.mainloop()


def save_results():
    global results
    def map0(x):
        xv = eval(x)
        if abs(xv) == 0:
            xv = 0
        return xv

    results["length"] = results["length"].map("{:.3f}".format).map(eval)
    results["strength"] = results["strength"].map("{:.3f}".format).map(map0)

    results.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def load_results(fp=filepath):
    try:
        return pd.read_csv(fp)
    except FileNotFoundError as e:
        print(e.strerror)
        return pd.DataFrame(columns=['strength', 'length', 'spike_probability'])


results = load_results()
gui_thread = threading.Thread(target=gui_loop, daemon=True)
gui_thread.start()
run_remaining()
save_results()
print(results)

plot_pulse_shape_spike_probability(results)

