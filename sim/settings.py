from dataclasses import dataclass, field


@dataclass
class NoiseSettings:
    frequency_hz: float = 0.35
    octaves: int = 3
    lacunarity: float = 2.0
    persistence: float = 0.55
    rotation_amplitude_deg: tuple[float, float, float] = (8.0, 7.0, 12.0)
    translation_amplitude_m: tuple[float, float, float] = (0.025, 0.02, 0.015)
    seed: int = 12


@dataclass
class JerkSettings:
    mean_rate_hz: float = 0.8
    rotation_range_deg: tuple[float, float] = (3.0, 10.0)
    translation_range_m: tuple[float, float] = (0.003, 0.018)
    rise_range_s: tuple[float, float] = (0.008, 0.025)
    decay_range_s: tuple[float, float] = (0.07, 0.20)
    seed: int = 42


@dataclass
class FocusSettings:
    distance_m: float = 3.2
    blur_strength: float = 3.0
    max_sigma_px: float = 6.0


@dataclass
class AppSettings:
    target_pitch_rad: float = 0.0
    target_yaw_rad: float = 0.0
    window_size: tuple[int, int] = (1440, 620)
    render_fps: float = 60.0
    home_position: tuple[float, float, float] = (0.0, 0.0, 1.2)
    noise: NoiseSettings = field(default_factory=NoiseSettings)
    jerks: JerkSettings = field(default_factory=JerkSettings)
    focus: FocusSettings = field(default_factory=FocusSettings)
