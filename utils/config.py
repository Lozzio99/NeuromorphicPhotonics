import math

from numpy import multiply, sqrt
from numpy.random import normal

# simulation parameters
t0 = 0
tf = 5e3
dt = 0.1


# LASER MODEL
# system parameters
k = 0.7
A = 1 / k
alpha = 2
h = 4

# timescale variables
gamma = 0.004
epsilon = 0.0001

# delta system control variable
DELTA_OFF = 0.95
DELTA_ALTERNATE = 1.05
INITIAL_DELTA = DELTA_OFF
INITIAL_STATE = lambda delta: [0, INITIAL_DELTA, (1 - INITIAL_DELTA) / k]

FIXED_DELTA = lambda t: INITIAL_DELTA

def sinusoidal_delta(t, periods=1.5, min_delta=1.01, max_delta=1.49):
    # Map time range -> sine -> positive sine -> delta range
    time = (math.sin((t/tf) * periods * 2 * math.pi) + 1) / 2.0
    return min_delta + time * (max_delta - min_delta)

def rectangle_spikes_delta(t, first=1000, interval=1000, duration=200, min_value=DELTA_OFF, max_value=DELTA_ALTERNATE):
    spike_starts = [first, first+duration+interval]
    for spike in spike_starts:
        t_max = spike + duration
        if spike <= t < t_max:
            return max_value
    return min_value

def rectangle_spikes_pulse(t, first=1000, interval=1000, duration=200, min_value=0, max_value=0.05):
    spike_starts = [first, first + duration + interval]
    for spike in spike_starts:
        t_max = spike + duration
        if spike <= t < t_max:
            return max_value
    return min_value

# External pulse for logarithmic amplifier
PULSE_OFF = lambda t: 0

# Noise complex
# sigma = 1 / (510 ** 2)
sigma = 1e-3
GAUSSIAN_NOISE_3D = lambda: multiply(sigma, [normal(0, sqrt(dt)), 0, 0])

# logarithmic amplifier function
GXf = lambda x, pt: A * math.log(1 + (alpha * (x + pt)))
nGXf = lambda x, pt: A * math.log(1 + (alpha * x)) + pt

# electrical field magnitude function
Xf = lambda e: (abs(e) ** 2)


## LIF MODEL
sustained_spike_mode = False
RESULTS_DIRECTORY = '../results/'