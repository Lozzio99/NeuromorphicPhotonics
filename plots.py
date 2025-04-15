from matplotlib import pyplot as plt

def plot_single_laser_solution(solution:dict[object, list[float]]):
    # plot x,y,w over time
    plt.figure()

    ts = solution['t']
    plt.plot(ts, solution['x'], 'r')
    plt.plot(ts, solution['y'], 'g')
    plt.plot(ts, solution['w'], 'b')

    plt.xlabel('t')

    plt.legend(["x(t)", "y(t)", "w(t)"], loc='upper right')
    plt.suptitle("Single Laser System Variables Solution")

    plt.tight_layout()
    plt.show()
