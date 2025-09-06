from classes.laser import LaserSystem
from classes.neuron import LIFNeuron
from utils import runner
from utils.plots import load_and_plot, plot_single_laser_solution, plot_neuron_solution_pulse, plot_neuron_solution
from utils.runner import store_solution_data

t1 = 500
t2 = 600
t3 = 700

p = 0.35

pos_pulse = lambda t: p if t1 < t < t2 else 0
neg_pulse = lambda t: -p if t1 < t < t2 else 0
seq_pulse = lambda t: -p if t1 < t < t2 else p if t2 < t < t3 else 0

single_neuron = LIFNeuron(pulse_f=pos_pulse)
test_simulation = runner.run_single_system_simulation(single_neuron)
plot_neuron_solution_pulse(test_simulation)

single_neuron = LIFNeuron(pulse_f=neg_pulse)
test_simulation = runner.run_single_system_simulation(single_neuron)
plot_neuron_solution_pulse(test_simulation)

single_neuron = LIFNeuron(pulse_f=seq_pulse)
test_simulation = runner.run_single_system_simulation(single_neuron)
plot_neuron_solution_pulse(test_simulation)