from typing import Any

import numpy as np
from numpy import ndarray, dtype
from numpy.random import normal

from classes.laser import LaserSystem
from utils.config import FIXED_DELTA, PULSE_OFF, GXf, gamma, epsilon, k, Xf, h, INITIAL_STATE, t0


class NIrisLaser(LaserSystem):
    def __init__(self, n_iris, delta_f=FIXED_DELTA, pulse_f=PULSE_OFF, amplifier_f=GXf, method='mean'):
        assert n_iris > 0
        assert method in ('mean', 'sum')
        self.n_iris = n_iris
        self.method = method

        self.eN = []
        self.yN = []
        self.wN = []
        self.x = None
        self.delta_f = delta_f
        self.deltas = [self.delta_f(t0) + normal(0, 0.01) for _ in range(n_iris)]
        self.initial_state = np.array([INITIAL_STATE(self.deltas[i]) for i in range(n_iris)])
        super().__init__(self.initial_state, delta_f, pulse_f, amplifier_f)

        # override
        self.gaussian3d = self.noise_f
        self.noise_f = lambda: np.array([self.gaussian3d() for _ in range(self.n_iris)])


    def update(self, state:ndarray, t:float):
        self.eN = [state[x, 0] for x in range(self.n_iris)]
        self.yN = [state[x, 1] for x in range(self.n_iris)]
        self.wN = [state[x, 2] for x in range(self.n_iris)]
        self.x = self.X(state)
        self.pulse = self.pulse_f(t)


    def get_state_dict(self) -> dict:
        state_dict = {
            'pulse': self.pulse,
        }

        for i in range(self.n_iris):
            state_dict[f'delta{i}'] = self.deltas[i]
            state_dict[f'e{i}'] = self.eN[i]
            state_dict[f'x{i}'] = Xf(self.eN[i]).real
            state_dict[f'y{i}'] = self.yN[i].real
            state_dict[f'w{i}'] = self.wN[i].real


        state_dict['x'] = self.x
        return state_dict


    def get_sysvar(self):
        return np.array([[self.eN[i], self.yN[i], self.wN[i]] for i in range(self.n_iris)])

    def reset(self):
        self.deltas = [self.delta_f(t0) + normal(0, 0.01) for _ in range(self.n_iris)]
        self.update(np.array([INITIAL_STATE(self.deltas[i]) for i in range(self.n_iris)]), t0)


    def X(self, state:ndarray) -> float:
        xs = 0
        for i in range(self.n_iris):
            xs += Xf(state[i, 0])
        return xs / self.n_iris if self.method == 'mean' else xs

    def rate_equations(self, state:ndarray, t=None) -> ndarray:
        # System Equations for the rate of change

        x = self.X(state)
        pt = self.pulse
        gx = self.ampli_f(x, pt)
        w_dot = -epsilon * (state[0, 2] + gx)
        s_dot = np.empty(shape=(self.n_iris, 3), dtype=complex)

        for i in range(self.n_iris):
            e,y,w = state[i]
            x = Xf(e)
            xy = x * y
            s_dot[i] = [
                0.5 * complex(1, h) * e * (y - 1),
                gamma * (self.deltas[i] - y + k * (w + gx) - xy),
                w_dot
            ]
        return s_dot