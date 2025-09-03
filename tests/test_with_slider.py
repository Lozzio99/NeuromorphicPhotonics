import tkinter as tk

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from classes.neuron import LIFNeuron
from utils import runner
from utils.config import DELTA_OFF, DELTA_ALTERNATE

# Main Tkinter window
root = tk.Tk()
root.title('Rectangle Spikes Function')
root.geometry("650x650")


fig, ax = plt.subplots(3, 1, sharex=True, height_ratios=[0.7, 0.15, 0.15])
canvas = FigureCanvasTkAgg(fig, master=root)
toolbar = NavigationToolbar2Tk(canvas, root)
canvas.get_tk_widget().pack()
toolbar.update()
toolbar.pack()

interval_var = tk.IntVar(value=1000)  # Default value

test = 'pulse'

def plot_neuron_solution(solution: dict[object, list[float]]):
    for axi in ax:
        axi.clear()

    ts = solution['t']
    ax[0].plot(ts, solution['x'], 'r')
    ax[0].plot(ts, solution['y'], 'g')
    ax[0].plot(ts, solution['w'], 'b')

    ax[0].hlines(1.0, solution['t'][0], solution['t'][-1], 'k', '--')

    ax[0].set_xlabel("t")

    ax[0].legend(["x(t)", "y(t)", "w(t)"], loc='upper right')
    ax[0].set_title('Single Laser System Variables Solution')

    if test == 'pulse':
        ax[1].plot(ts, solution['pulse'], 'k')
        ax[1].set_title(f"Pulse value")
    elif test == 'delta':
        ax[1].plot(ts, solution['delta'], 'k')
        ax[1].set_title(f"Delta value")
    else:
        raise Exception('test var not set correctly.')

    ax[2].plot(ts, solution['binary_state'], 'r')
    ax[2].set_title("Binary Output")

    plt.tight_layout()
    canvas.draw()

## delta/pulse : (500, 1500, 100)
## delta-sustained: (1500, 2500, 100)
## pulse-sustained: (1500, 2500, 100)
start = 200
end = 1200
res = 100
intervals = [i for i in range(start, end+1, res)]
solutions = {}

for interval in intervals:

    def rectangle_spikes_delta(t, iv=interval, first=1000, length=200, miv=DELTA_OFF, mav=DELTA_ALTERNATE):
        spike_starts = [first, first + length + iv]
        for spike in spike_starts:
            t_max = spike + length
            if spike <= t < t_max:
                return mav
        return miv

    def rectangle_spikes_pulse(t, iv=interval, first=1000, length=200, miv=0, mav=0.02):
        spike_starts = [first, first + length + iv]
        for spike in spike_starts:
            t_max = spike + length
            if spike <= t < t_max:
                return mav
        return miv

    if test == 'pulse':
        single_laser_test = LIFNeuron(pulse_f=rectangle_spikes_pulse)
    else :
        single_laser_test = LIFNeuron(delta_f=rectangle_spikes_delta)

    test_simulation = runner.run_single_system_simulation(single_laser_test)
    solutions[interval] = test_simulation

def plot(interval_value):
    simulation =  solutions[interval_value]
    plot_neuron_solution(simulation)

def on_interval_change(val):
    plot(int(val))

slider = tk.Scale(root, from_=start, to=end, resolution=res, orient=tk.HORIZONTAL, variable=interval_var, label="Interval", command=on_interval_change)
slider.pack(fill=tk.X, padx=20, pady=20)

plot(interval_var.get())
root.mainloop()
