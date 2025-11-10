import numpy as np
from numpy import ndarray
from classes.laser import LaserSystem
from classes.multi_iris_laser import NIrisLaser
from utils.config import Xf, sustained_spike_mode, PULSE_OFF, FIXED_DELTA


class LIFNeuron(NIrisLaser):
    def __init__(self, threshold=1.0, delta_f=FIXED_DELTA, pulse_f=PULSE_OFF, s_spk=sustained_spike_mode, n_iris=1):
        # LIF core1
        self.threshold = threshold
        self.spike_start = -np.inf
        self.spike = False
        self.spike_history = []

        # Sustained spike behavior
        self.sustained_spike_mode = s_spk
        super().__init__(delta_f=delta_f, pulse_f=pulse_f, n_iris=n_iris)


    def update(self, state: ndarray, t: float):
        super().update(state, t)
        self._verify_spiking(t)

    def _emit_spike(self, t: float):
        if not self.spike:
            self.spike = True
            self.spike_start = t

        # either way,
        self.spike_history.append(t)


    def _verify_spiking(self, t: float):
        if self.x >= self.threshold:
            # record spike
            self._emit_spike(t)
            return
        else:
            if self.sustained_spike_mode and self.spike:
                if self.x > 1e-4:
                    self._emit_spike(t)
                    return
            self.spike = False
            self.spike_start = -np.inf


    def get_state_dict(self) -> dict:
        base_state = super().get_state_dict()
        base_state['binary_state'] = int(self.spike)
        base_state['spike_start'] = self.spike_start if self.spike else -np.inf
        return base_state

    def reset(self):
        super().reset()
        self.spike = False
        self.spike_start = -np.inf
        self.spike_history.clear()
