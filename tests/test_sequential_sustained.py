from classes.neuron import LIFNeuron
from utils import runner
from utils.config import rectangle_spikes_delta
from utils.plots import load_and_plot, plot_neuron_solution_delta
from utils.runner import store_solution_data

filename = 'test_sequential_spikes_delta.csv'

single_laser_test = LIFNeuron(delta_f=rectangle_spikes_delta)
test_simulation = runner.run_single_system_simulation(single_laser_test)
store_solution_data(test_simulation, filename=filename)

load_and_plot(filename=filename, plot_f=plot_neuron_solution_delta)
