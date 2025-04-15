import math
from numpy import multiply, sqrt
from numpy.random import normal
import config

# laser parameters
k = 0.7
A = 1 / k
alpha = 2
h = 4

# timescale variables
gamma = 0.004
epsilon = 0.0001


# initial_state, around OFF fixed point
DELTA_OFF = 1.6
INITIAL_STATE = {
    'e': 0,
    'y': DELTA_OFF,
    'w': (1-DELTA_OFF)/k
}

# Noise complex
sigma = 1 / (510 ** 2)
GAUSSIAN_NOISE_3D = lambda: multiply(sigma, [normal(0, sqrt(config.dt)), 0, 0])

# logarithmic amplifier function
GXf = lambda x, pt: A * math.log(1 + (alpha * (x + pt)))

# electrical field magnitude function
Xf = lambda e: (abs(e) ** 2).real


class LaserSystem:

    def __init__(self, initial_state=None, delta_function=None, pulse_function=None, noise_function=None):
        if initial_state is None:
            initial_state = INITIAL_STATE

        self.e = initial_state['e']       # complex component of the electrical field
        self.x = Xf(self.e)               # x system variable
        self.y = initial_state['y']       # y system variable
        self.w = initial_state['w']       # w system variable

        self.pulse_f = pulse_function if pulse_function is not None else lambda t: 0
        self.delta_f = delta_function if delta_function is not None else lambda t: DELTA_OFF
        self.noise_f = noise_function if noise_function is not None else GAUSSIAN_NOISE_3D

        self.delta = self.delta_f(t=config.t0)
        self.pulse = self.pulse_f(t=config.t0)


    def get_state_dict(self) -> dict:
        return {
            'e': self.e,
            'x': self.x,
            'y': self.y,
            'w': self.w,
            'delta': self.delta,
            'pulse': self.pulse,
        }

    def get_sysvar(self):
        return [self.e, self.y, self.w]

    def update(self, state:list[float], t:float):
        self.e = state[0]
        self.y = state[1]
        self.w = state[2]
        self.x = Xf(self.e)
        self.delta = self.delta_f(t)
        self.pulse = self.pulse_f(t)


    def rate_equations(self, state:list[float], t) -> list[complex | float]:
        # System Equations for the rate of change

        # system variables
        e = state[0]
        y = state[1]
        w = state[2]

        x = Xf(e)
        xy = x * y
        pt = self.pulse
        gx = GXf(x, pt)
        delta = self.delta

        # rate equations
        e_dot = 0.5 * complex(1, 4) * e * (y - 1)
        y_dot = gamma * (delta - y + k * (w + gx) - xy)
        w_dot = -epsilon * (w + gx)

        return [e_dot, y_dot, w_dot]
