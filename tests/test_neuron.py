from classes.neuron import LIFNeuron
from utils import runner
from utils.plots import load_and_plot, plot_neuron_solution
from utils.runner import store_solution_data

filename = f'neuron_delta_sin_high_noise.csv'

neuron = LIFNeuron(threshold=1.05)

test_simulation = runner.run_single_system_simulation(neuron)
store_solution_data(test_simulation, filename=filename)

load_and_plot(filename=filename, plot_f=plot_neuron_solution)
