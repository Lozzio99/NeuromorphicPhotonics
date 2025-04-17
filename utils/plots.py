import pandas as pd
from matplotlib import pyplot as plt
import csv

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

def load_and_plot(filename, plot_f, directory=RESULTS_DIRECTORY):
    loaded = pd.read_csv(directory + filename)
    reshaped = loaded.to_dict('list')
    plot_f(reshaped)
