from utils import runner
from classes.laser import LaserSystem
from utils.config import DELTA_OFF
from utils.plots import plot_single_laser_solution, load_and_plot
from utils.runner import store_solution_data

single_laser_test = LaserSystem()
test_simulation = runner.run_single_laser_simulation(single_laser_test)

filename = f'single_laser_test_delta_{DELTA_OFF}.csv'
store_solution_data(test_simulation, filename=filename)

# plot_single_laser_solution(test_simulation)
load_and_plot(filename=filename, plot_f=plot_single_laser_solution)


