import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from utils.config import RESULTS_DIRECTORY, t0, tf, GIFS_DIRECTORY, INITIAL_DELTA, k, A, alpha


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

def plot_neuron_solution(solution:dict[object, list[float]], input_val:str):
    assert input_val is not None

    ts = solution['t']
    fig, ax = plt.subplots(3, 1, sharex=True, height_ratios=[0.7, 0.15, 0.15])

    ax[0].plot(ts, solution['x'], 'r')
    ax[0].plot(ts, solution['y'], 'g')
    ax[0].plot(ts, solution['w'], 'b')

    ax[0].set_xlabel("t")

    ax[0].legend(["x(t)", "y(t)", "w(t)"], loc='upper right')
    ax[0].set_title('Single Laser System Variables Solution')

    ax[1].plot(ts, solution[input_val], 'k')
    ax[1].set_title(f"{input_val.capitalize()} value")

    ax[2].plot(ts, solution['binary_state'], 'r')
    ax[2].set_title("Binary Output")

    plt.tight_layout()
    plt.show()

def plot_neuron_solution_pulse(solution:dict[object, list[float]]):
    plot_neuron_solution(solution, input_val='pulse')

def plot_neuron_solution_delta(solution:dict[object, list[float]]):
    plot_neuron_solution(solution, input_val='delta')

def load_and_plot(filename, plot_f, directory=RESULTS_DIRECTORY):
    loaded = pd.read_csv(directory + filename)
    reshaped = loaded.to_dict('list')
    plot_f(reshaped)

def plot_pulse_shape_spike_probability(df):
    pivot_df = df.pivot(index='strength', columns='length', values='spike_probability')

    Y, X = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        vmin=0, vmax=1,  # force colorbar limits between 0 and 1
        linewidth=0, antialiased=True
    )

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Spike Probability (0-1)')

    # Labels
    ax.set_ylabel('Pulse Length')
    ax.set_xlabel('Pulse Strength')
    ax.set_zlabel('Spike Probability')

    ax.view_init(elev=30, azim=260)

    plt.tight_layout()
    plt.show()

def plot_neuron_solution_xw(solution:dict[object, list[float]], ax=None):
    if ax is None:
        plt.plot(solution['x'], solution['w'],'k')
        plt.xlabel('x'); plt.ylabel('w')
        plt.show()
    else:
        ax.plot(solution['x'], solution['w'], 'k')
        ax.set_xlabel('x'); ax.set_ylabel('w')

def plot_neuron_solution_xy(solution: dict[object, list[float]], ax=None):
    if ax is None:
        plt.plot(solution['x'], solution['y'], 'k')
        plt.xlabel('x'); plt.ylabel('y')
        plt.show()
    else:
        ax.plot(solution['x'], solution['y'], 'k')
        ax.set_xlabel('x'); ax.set_ylabel('y')

def plot_neuron_solution_yw(solution: dict[object, list[float]], ax=None):
    if ax is None:
        plt.plot(solution['y'], solution['w'], 'k')
        plt.xlabel('y'); plt.ylabel('w')
        plt.show()
    else:
        ax.plot(solution['y'], solution['w'], 'k')
        ax.set_xlabel('y'); ax.set_ylabel('w')

def plot_phase_space(solution:dict[object, list[float]]):
    fig, axes = plt.subplots(1, 3)

    plot_neuron_solution_xy(solution, axes[0])
    plot_neuron_solution_xw(solution, axes[1])
    plot_neuron_solution_yw(solution, axes[2])

    plt.tight_layout()
    plt.show()

def scale_bounds(a, b, f=0.1):
    width = b - a
    pad = f * width
    return a - pad, b + pad

def plot_phase_space_dynamical(solution:dict[object, list[float]], gif_name=None):
    frames = 100

    ts = solution['t']
    x = solution['x']
    y = solution['y']
    w = solution['w']
    p = solution['pulse']

    x1 = ts[0]
    x2 = ts[len(ts) - 1]
    t_span = round((x2 - x1) / frames)

    # fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :])

    l0, = ax0.plot([], [], 'k', linewidth=2)
    l1, = ax1.plot([], [], 'k', linewidth=2)
    l2, = ax2.plot([], [], 'k', linewidth=2)
    l3, = ax3.plot([], [], 'k', linewidth=2)



    xl = scale_bounds(min(x), max(x))
    yl = scale_bounds(min(y), max(y))
    wl = scale_bounds(min(w), max(w))
    pl = scale_bounds(min(min(p), -max(p)), max(max(p), -min(p)))
    tl = scale_bounds(t0, tf, 0.001)

    ax0.set_xlim(xl); ax0.set_ylim(yl)
    ax1.set_xlim(xl); ax1.set_ylim(wl)
    ax2.set_xlim(yl); ax2.set_ylim(wl)
    ax3.set_xlim(tl); ax3.set_ylim(pl)

    ax0.set_xlabel('x'); ax0.set_ylabel('y')
    ax1.set_xlabel('x'); ax1.set_ylabel('w')
    ax2.set_xlabel('y'); ax2.set_ylabel('w')
    ax3.set_xlabel('t'); ax3.set_ylabel('pt')

    plt.tight_layout()

    writer, output_file = make_gif_writer(gif_name)

    with writer.saving(fig, output_file, frames):
        t = t0
        for i in range(0, frames - 1):
            if i % 20 == 0:
                print(f"frame {i} / {frames}")
            ftf = min(t + t_span, tf)
            ft0 = max(0, ftf - (t_span * 40))

            ida = ts.index(ft0)
            idz = ts.index(ftf)

            xt = x[ida:idz]
            yt = y[ida:idz]
            wt = w[ida:idz]
            pt = p[ida:idz]
            tt = ts[ida:idz]

            l0.set_data(xt, yt)
            l1.set_data(xt, wt)
            l2.set_data(yt, wt)
            l3.set_data(tt, pt)

            writer.grab_frame()
            t = ftf

    print(f"Gif file {output_file.absolute()} successfully created.")

    l0.set_data(x, y)
    l1.set_data(x, w)
    l2.set_data(y, w)
    l3.set_data(ts, p)

    plt.tight_layout()
    plt.show()

def off_manifold_lines(w_range=(-3, 1), resolution=200):
    fp = (1 - INITIAL_DELTA) / k
    w_stable = np.linspace(w_range[0], fp, resolution)
    w_unstable = np.linspace(fp, w_range[1], resolution)

    x_unstable = w_unstable * 0
    x_stable = w_stable * 0

    y_fn = lambda w: INITIAL_DELTA + (k * w)

    y_unstable = y_fn(w_unstable)
    y_stable = y_fn(w_stable)

    return (x_unstable, y_unstable, w_unstable), (x_stable, y_stable, w_stable)

def on_manifold_lines(x_range=(0, 5), resolution=200):
    fp = k * A - (1 / alpha)

    print(fp)
    x_unstable = np.linspace(x_range[0], fp, resolution)
    x_stable = np.linspace(fp, x_range[1], resolution)

    y_unstable = [1 for _ in x_unstable]
    y_stable = [1 for _ in x_stable]

    w_fn = lambda x: ((x - INITIAL_DELTA + 1) / k) - (A * math.log(1 + (alpha * x)))

    w_unstable = [w_fn(x) for x in x_unstable]
    w_stable = [w_fn(x) for x in x_stable]

    return (x_unstable, y_unstable, w_unstable), (x_stable, y_stable, w_stable)

def plot_neuron_solution_3d_dynamical(solution:dict[object, list[float]]):
    t = solution['t']
    x = solution['x']
    y = solution['y']
    w = solution['w']
    p = solution['pulse']

    fig = plt.figure(figsize=(9, 7))
    gs = GridSpec(2, 1, height_ratios=[75, 25])
    ax3d = fig.add_subplot(gs[0], projection='3d')

    ax3d.view_init(elev=30, azim=-30)

    ax2d = fig.add_subplot(gs[1])

    x_min, x_max = scale_bounds(np.min(x), np.max(x), 0.1)
    y_min, y_max = scale_bounds(np.min(y), np.max(y), 0.1)
    w_min, w_max = scale_bounds(np.min(w), np.max(w), 0.1)

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("w")

    line3d, = ax3d.plot([], [], [], lw=2, color="black")
    tracer3d, = ax3d.plot([], [], [], "ro")

    mnf_off_unstable, = ax3d.plot([], [], [], "r--", lw=1)
    mnf_off_stable, = ax3d.plot([], [], [], "b-", lw=1)
    off_unstable, off_stable = off_manifold_lines(w_range=(w_min, w_max))

    mnf_on_unstable, = ax3d.plot([], [], [], "r--", lw=1)
    mnf_on_stable, = ax3d.plot([], [], [], "b-", lw=1)
    on_unstable, on_stable = on_manifold_lines(x_range=(0, x_max))


    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(w_min, w_max)

    ax2d.plot(t, p, "k-", lw=1)
    tracer2d, = ax2d.plot([], [], "ro")

    ax2d.set_xlabel("t")
    ax2d.set_ylabel("p(t)")
    ax2d.set_xlim(np.min(t), np.max(t))
    p_min, p_max = scale_bounds(np.min(p), np.max(p), 0.1)
    ax2d.set_ylim(p_min, p_max)

    trail_length = int(1e6)

    def update(frame):
        idx = frame

        start = max(0, idx - trail_length)
        line3d.set_data(x[start:idx], y[start:idx]); line3d.set_3d_properties(w[start:idx])

        tracer3d.set_data(x[idx:idx + 1], y[idx:idx + 1]); tracer3d.set_3d_properties(w[idx:idx + 1])

        mnf_off_unstable.set_data(off_unstable[0], off_unstable[1]); mnf_off_unstable.set_3d_properties(off_unstable[2])
        mnf_off_stable.set_data(off_stable[0], off_stable[1]); mnf_off_stable.set_3d_properties(off_stable[2])

        mnf_on_unstable.set_data(on_unstable[0], on_unstable[1]); mnf_on_unstable.set_3d_properties(on_unstable[2])
        mnf_on_stable.set_data(on_stable[0], on_stable[1]); mnf_on_stable.set_3d_properties(on_stable[2])

        ax3d.set_title(f"t = {t[idx]:.2f}")

        tracer2d.set_data([t[idx]], [p[idx]])

        return line3d, tracer3d, tracer2d

    ani = FuncAnimation(fig, update, frames=range(0, len(t), 50), interval=50, blit=False)

    plt.show()


def make_gif_writer(gif_name):
    writer = PillowWriter(fps=15)
    fn = os.path.join(GIFS_DIRECTORY, f"phase_space.gif" if gif_name is None else gif_name)
    output_file = Path(fn)

    print(f"Creating gif file in: {output_file.absolute()}")
    return writer, output_file