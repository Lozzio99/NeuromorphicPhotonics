from classes.neuron import LIFNeuron
from utils import runner
from utils.plots import  plot_neuron_solution_pulse
from utils.runner import store_solution_data

n_iris = 4
# multi_iris_laser = NIrisLaser(n_iris=n_iris, pulse_f=lambda t: -0.2 if 2000<=t<=2100 else 0)
# test_simulation = runner.run_single_system_simulation(multi_iris_laser)
# filename = f'test_{n_iris}iris_laser.csv'
# store_solution_data(test_simulation, filename=filename)
#
# plot_multi_iris_laser_solution(test_simulation)


multi_iris_neuron = LIFNeuron(n_iris=n_iris, pulse_f=lambda t: 0 if 2000<=t<=2100 else 0)
test_simulation = runner.run_single_system_simulation(multi_iris_neuron)
filename = f'test_{n_iris}iris_neuron.py'
store_solution_data(test_simulation, filename)

plot_neuron_solution_pulse(test_simulation)
