import math

from classes.neuron import LIFNeuron
from utils import runner
from utils.config import A, alpha, DELTA_ALTERNATE
from utils.plots import *

t1 = 1000
t2 = 1100
t3 = 1300

p = 0.3

pos_pulse = lambda t: p if t1 < t < t2 else  0
neg_pulse = lambda t: -p if t2 < t < t3 else  0
double_pulse = lambda t: p if t1 < t < t2 else -p if t2 < t < t3 else 0
double_reverse = lambda t: -p if t1 < t < (t1 + (t3-t2)) else p if (t1 + (t3-t2)) < t < (t1 + (t3-t2)) + (t2 - t1) else 0

single_neuron = LIFNeuron(pulse_f=pos_pulse)
test_simulation = runner.run_single_system_simulation(single_neuron)


# plot_neuron_solution_pulse(test_simulation)

# plot_neuron_solution_xw(test_simulation)
# plot_neuron_solution_xy(test_simulation)
# plot_neuron_solution_yw(test_simulation)

# plot_phase_space(test_simulation)
# plot_phase_space_dynamical(test_simulation, 'double_pulse_phase_space.gif')


plot_neuron_solution_3d_dynamical(test_simulation)

