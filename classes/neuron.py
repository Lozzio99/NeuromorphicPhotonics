import numpy as np

from classes.laser import LaserSystem
from utils.config import Xf, sinusoidal_delta, INITIAL_STATE, spike_mode, SUSTAINED_SPIKE_DURATION


class LIFNeuron(LaserSystem):
    def __init__(self, threshold=1.0, input_fn=sinusoidal_delta):
        super().__init__()
        # LIF core
        self.threshold = threshold
        self.last_spike_time = -np.inf
        self.spike = False
        self.spike_history = []

        # Sustained spike behavior
        if spike_mode == 'sustained':
            self.sustained_spike_mode = True
            self.sustained_duration = SUSTAINED_SPIKE_DURATION
            self._sustained_spike_end = -np.inf

        else:
            self.sustained_spike_mode = False

        self.delta_f = input_fn
        self.update(INITIAL_STATE(self.delta_f(t=0)), t=0)

    def update(self, state: list[complex], t: float):
        super().update(state, t)
        if self._verify_sustained_spiking(t): return
        self._verify_spiking(t)

    def _emit_spike(self, t: float):
        self.spike = True
        self.last_spike_time = t
        self.spike_history.append(t)

    def _verify_sustained_spiking(self, t: float):
        # Handle sustained spike for when it's still ongoing
        if self.sustained_spike_mode and self.spike:
            # if still ongoing
            if t <= self._sustained_spike_end:
                self._emit_spike(t)
                return True
            # else interrupt
            else:
                self.spike = False
        return False

    def _verify_spiking(self, t: float):
        self.spike = False
        # Spike trigger condition upon threshold
        if Xf(self.e) > self.threshold:
            if self.sustained_spike_mode:
                # Start the sustained spike if it hasn't yet
                self._sustained_spike_end = t + self.sustained_duration
            # record spike
            self._emit_spike(t)

    def get_state_dict(self) -> dict:
        base_state = super().get_state_dict()
        base_state['binary_state'] = int(self.spike)
        base_state['last_spike_time'] = self.last_spike_time
        return base_state

    def reset(self):
        self.update(INITIAL_STATE(self.delta_f(t=0)), t=0)
        self.spike = False
        self.last_spike_time = -np.inf
        self.spike_history.clear()
