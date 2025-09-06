import numpy as np

from classes.laser import LaserSystem
from utils.config import Xf, INITIAL_STATE, sustained_spike_mode, PULSE_OFF, FIXED_DELTA


class LIFNeuron(LaserSystem):
    def __init__(self, threshold=1.0, delta_f=FIXED_DELTA, pulse_f=PULSE_OFF, s_spk=sustained_spike_mode):

        # LIF core1
        self.threshold = threshold
        self.spike_start = -np.inf
        self.spike = False
        self.spike_history = []

        # Sustained spike behavior
        self.sustained_spike_mode = s_spk
        super().__init__(delta_f=delta_f, pulse_f=pulse_f)


    def update(self, state: list[complex], t: float):
        super().update(state, t)
        self._verify_spiking(t)

    def _emit_spike(self, t: float):
        if self.spike == False:
            self.spike = True
            self.spike_start = t

        # either way,
        self.spike_history.append(t)


    def _verify_spiking(self, t: float):
        if Xf(self.e) >= self.threshold:
            # record spike
            self._emit_spike(t)
            return
        else:
            if self.sustained_spike_mode and self.spike:
                if Xf(self.e) > 1e-4:
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
