import runner
from laser import LaserSystem
from plots import plot_single_laser_solution

single_laser_test = LaserSystem()
test_simulation = runner.run_single_laser_simulation(single_laser_test)
plot_single_laser_solution(test_simulation)