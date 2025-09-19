import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from utils.config import RESULTS_DIRECTORY


def plot_single_laser_solution(solution:dict[object, list[float]]):
    # plot x,y,w over time
    plt.figure()

    ts = solution['t']
    plt.plot(ts, solution['x'], 'r')
    plt.plot(ts, solution['y'], 'g')
    plt.plot(ts, solution['w'], 'b')

    plt.xlabel('t')

    plt.legend(["x(t)", "y(t)", "w(t)"], loc='upper right')
    plt.suptitle("Single Laser System Variables Solution")

    plt.tight_layout()
    plt.show()


def plot_neuron_solution(solution:dict[object, list[float]], input_val:str):
    assert input_val is not None

    ts = solution['t']
    fig, ax = plt.subplots(3, 1, sharex=True, height_ratios=[0.7, 0.15, 0.15])

    ax[0].plot(ts, solution['x'], 'r')
    ax[0].plot(ts, solution['y'], 'g')
    ax[0].plot(ts, solution['w'], 'b')

    ax[0].set_xlabel("t")

    ax[0].legend(["x(t)", "y(t)", "w(t)"], loc='upper right')
    ax[0].set_title('Single Laser System Variables Solution')

    ax[1].plot(ts, solution[input_val], 'k')
    ax[1].set_title(f"{input_val.capitalize()} value")

    ax[2].plot(ts, solution['binary_state'], 'r')
    ax[2].set_title("Binary Output")

    plt.tight_layout()
    plt.show()


def plot_neuron_solution_pulse(solution:dict[object, list[float]]):
    plot_neuron_solution(solution, input_val='pulse')

def plot_neuron_solution_delta(solution:dict[object, list[float]]):
    plot_neuron_solution(solution, input_val='delta')

def load_and_plot(filename, plot_f, directory=RESULTS_DIRECTORY):
    loaded = pd.read_csv(directory + filename)
    reshaped = loaded.to_dict('list')
    plot_f(reshaped)


def plot_pulse_shape_spike_probability(df):
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
