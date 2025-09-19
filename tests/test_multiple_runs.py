from classes.neuron import LIFNeuron
from utils import runner

t1 = 500
t2 = 600

p = 0.35

pos_pulse = lambda t: p if t1 < t < t2 else 0

filename = 'multiple_runs_fixed_pulse.csv'
single_neuron = LIFNeuron(pulse_f=pos_pulse)
runs = runner.multiple_runs_simulation(single_neuron, n_runs=100, filter_vars=['t', 'pulse', 'binary_state'])
# store_solution_data(runs, filename=filename)

for i in runs:
    r = runs[i]
    print(sum(r['binary_state']))

