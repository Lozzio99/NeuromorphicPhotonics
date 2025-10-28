import os
import threading
import tkinter as tk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from classes.neuron import LIFNeuron
from utils import runner
from utils.config import RESULTS_DIRECTORY

t1 = 1000
length = 150
strength = 0.2

interval_range = np.arange(-250, 251, 5)
num_runs = 20

filepath = os.path.join(RESULTS_DIRECTORY, f'double_spike_interval_variation_{num_runs}_runs.csv')

def pulse_f(interval):
    pos_first = interval >= 0

    t1x = t1+length
    t2 = t1x + abs(interval)
    t2x = t2 + length

    if pos_first:
        p1 = strength
        p2 = -strength
    else:
        p1 = -strength
        p2 = strength


    return lambda t: p1 if t1 <= t < t1x else p2 if t2 <= t < t2x else 0

def load_results(fp=filepath):
    try:
        return pd.read_csv(fp)
    except FileNotFoundError as e:
        print(e.strerror)
        return pd.DataFrame(columns=['strength', 'length', 'interval', 'spike_probability'])

def save_results():
    global results
    def map0(x):
        xv = eval(x)
        if abs(xv) == 0:
            xv = 0
        return xv

    results["interval"] = results["interval"].map("{:.3f}".format).map(map0)
    results.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def gui_loop():
    root = tk.Tk()
    root.title("Save Results")
    root.geometry("200x100")

    save_btn = tk.Button(root, text="Save", command=save_results)
    save_btn.pack(expand=True)

    root.mainloop()

def run_remaining():
    global results

    for interval in interval_range:
        found = results.loc[(results['interval'] == interval)]

        if len(found) > 0:
            spike_probability = results.loc[(results['interval'] == interval)]['spike_probability'].values[0]
        else:
            single_neuron = LIFNeuron(pulse_f=pulse_f(interval))
            runs = runner.multiple_runs_simulation(single_neuron, n_runs=num_runs, filter_vars=['binary_state'])

            spikes = [1 if sum(runs[i]['binary_state']) > 0 else 0 for i in runs]
            spike_probability = sum(spikes) / len(spikes)

            results.loc[len(results)] = {
                'length': length,
                'strength':strength,
                'interval': interval,
                'spike_probability':spike_probability
            }

        print(length, strength, ' | ',interval, ' |\t:', spike_probability)


results = load_results()
gui_thread = threading.Thread(target=gui_loop, daemon=True)
gui_thread.start()
run_remaining()
save_results()
print(results)

x = results['interval']
y = results['spike_probability']

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Spike probability', color='black')

plt.axvline(0, color='gray', linestyle='--', linewidth=1)

plt.text(-200, max(y)*0.9,
         "Negative pulse\nbefore positive pulse",
         ha='center', va='top', fontsize=10, color='blue')

plt.text(200, max(y)*0.9,
         "Positive pulse\nbefore negative pulse",
         ha='center', va='top', fontsize=10, color='red')

plt.text(0, max(y)*0.95,
         "Switch point (0 ms)",
         ha='center', va='bottom', fontsize=9, color='gray')

# Axis labels and title
plt.xlabel("Interval between pulses")
plt.ylabel("Spike probability")
plt.title(f"Spike Probability vs. Interval Length, Pulse Length:{length}, Strength: {strength}")

# Optional visual cues
plt.fill_between(x, 0, y, where=(x < 0), color='blue', alpha=0.05)
plt.fill_between(x, 0, y, where=(x > 0), color='red', alpha=0.05)

plt.legend()
plt.tight_layout()
plt.show()

