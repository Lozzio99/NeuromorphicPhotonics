import numpy as np

from utils.config import INITIAL_STATE, GAUSSIAN_NOISE_3D, Xf, GXf, t0, gamma, epsilon, k, FIXED_DELTA, PULSE_OFF, nGXf


class LaserSystem:

    def __init__(self, initial_state=None, delta_f=FIXED_DELTA, pulse_f=PULSE_OFF, amplifier_f=GXf):
        self.pulse_f = pulse_f
        self.delta_f = delta_f
        self.ampli_f = amplifier_f

        if initial_state is None:
            initial_state = INITIAL_STATE(delta_f(t0))

        self.noise_f = GAUSSIAN_NOISE_3D

        self.e = None
        self.x = None
        self.y = None
        self.w = None
        self.delta = None
        self.pulse = None

        self.update(initial_state, t0)


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


    def reset(self):
        self.update(INITIAL_STATE(self.delta), t0)


    def rate_equations(self, state:list[float], t=None) -> list[complex | float]:
        # System Equations for the rate of change

        # system variables
        e = state[0]
        y = state[1]
        w = state[2]

        x = Xf(e)
        xy = x * y
        pt = self.pulse
        gx = self.ampli_f(x, pt)
        delta = self.delta

        # rate equations
        e_dot = 0.5 * complex(1, 4) * e * (y - 1)
        y_dot = gamma * (delta - y + k * (w + gx) - xy)
        w_dot = -epsilon * (w + gx)

        return [e_dot, y_dot, w_dot]
