import numpy as np

from classes.laser import LaserSystem
from utils.config import Xf, sinusoidal_delta, INITIAL_STATE, sustained_spike_mode, sustained_duration


class LIFNeuron(LaserSystem):
    def __init__(
        self,
        threshold=1.0,
        refractory_time=5.0,
        track_spike_history=True,
    ):

        super().__init__()
        self.delta_f = sinusoidal_delta
        self.update(INITIAL_STATE(self.delta_f(t=0)), t=0)

        # LIF core
        self.threshold = threshold
        self.refractory_time = refractory_time

        self.last_spike_time = -np.inf
        self.spike = False
        self.spike_history = [] if track_spike_history else None
        self.track_spike_history = track_spike_history

        # Sustained spike behavior
        self.sustained_spike_mode = sustained_spike_mode
        self.sustained_duration = sustained_duration
        self._sustained_spike_end = -np.inf

    def update(self, state: list[complex], t: float):
        super().update(state, t)
        self.spike = False


        # Handle sustained spike if still ongoing
        if self.sustained_spike_mode and t <= self._sustained_spike_end:
            self.spike = True
            return

        # Refractory check
        if t - self.last_spike_time < self.refractory_time:
            return

        # Spike trigger condition
        if np.abs(self.e) > self.threshold:
            if self.sustained_spike_mode:
                self.start_sustained_spike(t)
            else:
                self._emit_spike(t)


    def start_sustained_spike(self, t: float):
        # Extend if still spiking
        if t < self._sustained_spike_end:
            self._sustained_spike_end = t + self.sustained_duration
        else:
            self._sustained_spike_end = t + self.sustained_duration

        self.spike = True
        if self.track_spike_history:
            self.spike_history.append(t)

    def _emit_spike(self, t: float):
        self.spike = True
        self.last_spike_time = t

        if self.track_spike_history:
            self.spike_history.append(t)

    def is_spiking(self) -> bool:
        return self.spike

    def get_spike_times(self) -> list[float]:
        return self.spike_history if self.track_spike_history else []

