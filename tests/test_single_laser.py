from utils.config import DELTA_ALTERNATE
from utils.plots import plot_single_laser_solution, load_and_plot

# single_laser_test = LaserSystem()
# test_simulation = runner.run_single_system_simulation(single_laser_test)

filename = f'single_laser_test_delta_{DELTA_ALTERNATE}.csv'
# store_solution_data(test_simulation, filename=filename)

load_and_plot(filename=filename, plot_f=plot_single_laser_solution)


