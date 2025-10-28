import os

import pandas as pd

from utils.config import RESULTS_DIRECTORY
from utils.plots import plot_pulse_shape_spike_probability

fp1 = os.path.join(RESULTS_DIRECTORY, f'spike_probability_by_pulse_10_runs.csv')

data1 = pd.read_csv(fp1)

plot_pulse_shape_spike_probability(data1)
