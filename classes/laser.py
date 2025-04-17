import numpy as np

from utils.config import INITIAL_STATE, GAUSSIAN_NOISE_3D, Xf, GXf, t0, gamma, epsilon, k, FIXED_DELTA, PULSE_OFF


class LaserSystem:

    def __init__(self, initial_state=None):
        if initial_state is None:
            initial_state = INITIAL_STATE

        self.e = initial_state['e']       # complex component of the electrical field
        self.x = Xf(self.e)               # x system variable
        self.y = initial_state['y']       # y system variable
        self.w = initial_state['w']       # w system variable

        self.pulse_f = PULSE_OFF
        self.delta_f = FIXED_DELTA
        self.noise_f = GAUSSIAN_NOISE_3D

        self.delta = self.delta_f(t=t0)
        self.pulse = self.pulse_f(t=t0)


    def get_state_dict(self) -> dict:
        return {
            'e': self.e,
            'x': self.x.real,
            'y': self.y.real,
            'w': self.w.real,
            'delta': self.delta,
            'pulse': self.pulse,
        }

    def get_sysvar(self):
        return [self.e, self.y, self.w]

    def update(self, state:list[complex], t:float):
        state = np.real_if_close(np.array(state))
        self.e = state[0]
        self.y = state[1]
        self.w = state[2]
        self.x = Xf(self.e)
        self.delta = self.delta_f(t)
        self.pulse = self.pulse_f(t)


    def rate_equations(self, state:list[float], t=None) -> list[complex | float]:
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
