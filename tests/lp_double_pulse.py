import os

import pandas as pd

from utils.config import RESULTS_DIRECTORY
from utils.plots import plot_pulse_shape_spike_probability

fp2 = os.path.join(RESULTS_DIRECTORY, f'double_spike_100_runs.csv')

data2 = pd.read_csv(fp2)

plot_pulse_shape_spike_probability(data2)