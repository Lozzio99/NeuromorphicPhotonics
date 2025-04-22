import math
from numpy import multiply, sqrt
from numpy.random import normal

# simulation parameters
t0 = 0
tf = 2e5
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
DELTA_ALTERNATE = 1.1
INITIAL_DELTA = DELTA_ALTERNATE
INITIAL_STATE = lambda delta: [0, INITIAL_DELTA, (1 - INITIAL_DELTA) / k]

FIXED_DELTA = lambda t: INITIAL_DELTA

def sinusoidal_delta(t, periods=1.5, min_delta=1.01, max_delta=1.49):
    # Map time range -> sine -> positive sine -> delta range
    time = (math.sin((t/tf) * periods * 2 * math.pi) + 1) / 2.0
    return min_delta + time * (max_delta - min_delta)


# External pulse for logarithmic amplifier
PULSE_OFF = lambda t: 0

# Noise complex
# sigma = 1 / (510 ** 2)
sigma = 1e-3
GAUSSIAN_NOISE_3D = lambda: multiply(sigma, [normal(0, sqrt(dt)), 0, 0])

# logarithmic amplifier function
GXf = lambda x, pt: A * math.log(1 + (alpha * (x + pt)))

# electrical field magnitude function
Xf = lambda e: (abs(e) ** 2)


## LIF MODEL
spike_mode = 'sustained'
SUSTAINED_SPIKE_DURATION = 15e3

RESULTS_DIRECTORY = 'results/'