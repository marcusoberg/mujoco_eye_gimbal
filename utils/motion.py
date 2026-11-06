import math
from dataclasses import dataclass

import numpy as np


class Perlin1D:
    """Deterministic gradient noise; no third-party noise package."""

    def __init__(self, seed):
        self.seed = seed

    def gradient(self, lattice):
        value = (lattice * 374761393 + self.seed * 668265263) & 0xFFFFFFFF
        value = ((value ^ (value >> 13)) * 1274126177) & 0xFFFFFFFF
        return 1.0 if value & 1 else -1.0

    def sample(self, x):
        cell = math.floor(x)
        u = x - cell
        fade = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
        left = self.gradient(cell) * u
        right = self.gradient(cell + 1) * (u - 1.0)
        return 2.0 * ((1.0 - fade) * left + fade * right)


class SmoothMotion:
    """Independent fractal Perlin signals for XYZ and roll/pitch/yaw."""

    def __init__(self, settings):
        self.settings = settings
        self.channels = [Perlin1D(settings.seed + 31 * i) for i in range(6)]
        self.amplitudes = np.array([
            *settings.translation_amplitude_m,
            *np.deg2rad(settings.rotation_amplitude_deg),
        ])

    def sample(self, time_s):
        result = np.zeros(6)
        for index, noise in enumerate(self.channels):
            frequency = self.settings.frequency_hz
            amplitude = 1.0
            normalization = 0.0
            for _ in range(self.settings.octaves):
                result[index] += amplitude * noise.sample(time_s * frequency)
                normalization += amplitude
                frequency *= self.settings.lacunarity
                amplitude *= self.settings.persistence
            result[index] *= self.amplitudes[index] / normalization
        return result


@dataclass
class Pulse:
    start_s: float
    displacement: np.ndarray
    rise_s: float
    decay_s: float

    def sample(self, time_s):
        age = time_s - self.start_s
        envelope = (1.0 - math.exp(-age / self.rise_s)) * math.exp(-age / self.decay_s)
        return self.displacement * envelope


class RandomJerks:
    """Poisson-distributed, sharp-onset, exponentially decaying pose pulses."""

    def __init__(self, settings):
        self.settings = settings
        self.rng = np.random.default_rng(settings.seed)
        self.next_event_s = self.rng.exponential(1.0 / settings.mean_rate_hz)
        self.pulses = []

    def create_pulse(self, start_s):
        axis_pos = self.rng.normal(size=3)
        axis_pos /= np.linalg.norm(axis_pos)
        axis_rot = self.rng.normal(size=3)
        axis_rot /= np.linalg.norm(axis_rot)
        translation = self.rng.uniform(*self.settings.translation_range_m) * axis_pos
        rotation = math.radians(self.rng.uniform(*self.settings.rotation_range_deg)) * axis_rot
        return Pulse(
            start_s=start_s,
            displacement=np.r_[translation, rotation],
            rise_s=self.rng.uniform(*self.settings.rise_range_s),
            decay_s=self.rng.uniform(*self.settings.decay_range_s),
        )

    def sample(self, time_s):
        while time_s >= self.next_event_s:
            self.pulses.append(self.create_pulse(self.next_event_s))
            self.next_event_s += self.rng.exponential(1.0 / self.settings.mean_rate_hz)

        self.pulses = [
            pulse for pulse in self.pulses
            if time_s - pulse.start_s <= 8.0 * pulse.decay_s
        ]
        result = np.zeros(6)
        for pulse in self.pulses:
            result += pulse.sample(time_s)
        return result


class BaseMotion:
    """Produces [x, y, z, roll, pitch, yaw] offsets at simulation time t."""

    def __init__(self, noise_settings, jerk_settings):
        self.noise = SmoothMotion(noise_settings)
        self.jerks = RandomJerks(jerk_settings)

    def sample(self, time_s):
        return self.noise.sample(time_s) + self.jerks.sample(time_s)
