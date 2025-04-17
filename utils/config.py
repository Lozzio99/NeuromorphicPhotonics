import math
from numpy import multiply, sqrt
from numpy.random import normal

# simulation parameters
t0 = 0
tf = 1e5
dt = 0.1

# laser parameters
k = 0.7
A = 1 / k
alpha = 2
h = 4

# timescale variables
gamma = 0.004
epsilon = 0.0001

# initial_state, around OFF fixed point
DELTA_OFF = 1.1
INITIAL_STATE = {
    'e': 0,
    'y': DELTA_OFF,
    'w': (1-DELTA_OFF)/k
}

# Noise complex
sigma = 1 / (510 ** 2)
GAUSSIAN_NOISE_3D = lambda: multiply(sigma, [normal(0, sqrt(dt)), 0, 0])

# logarithmic amplifier function
GXf = lambda x, pt: A * math.log(1 + (alpha * (x + pt)))

# electrical field magnitude function
Xf = lambda e: (abs(e) ** 2)

RESULTS_DIRECTORY = 'results/'