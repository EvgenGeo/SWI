from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel
import numpy as np


@dataclass
class Seismogram:
    data: np.ndarray
    headers: Optional[np.ndarray]
    dt: float
    dx: float

    @property
    def f_s(self) -> float:
        """Частота дискретизации."""
        return 1 / self.dt

    @property
    def time_counts(self) -> int:
        """Количество отсчётов."""
        return np.shape(self.data)[0]

    @property
    def spatial_counts(self) -> int:
        """Количество трасс."""
        return np.shape(self.data)[1]


@dataclass
class Spectra:
    vf_spectra: np.ndarray
    velocities: np.ndarray
    frequencies: np.ndarray
    fk_spectra: np.ndarray
    wave_numbers: np.ndarray

    @property
    def d_vel(self) -> float:
        """Шаг по скорости."""
        return float(self.velocities[1] - self.velocities[0])

    @property
    def d_freq(self) -> float:
        """Шаг по частоте."""
        return float(self.frequencies[1] - self.frequencies[0])

    @property
    def d_k(self) -> float:
        """Шаг по волновому числу."""
        return float(self.wave_numbers[1] - self.wave_numbers[0])


# ___spectral___############
class SpectralAdvancedModel(BaseModel):
    desired_nt: int
    desired_nx: int
    smooth_data: bool
    width: float



class SpectralModel(BaseModel):
    fmin: float
    fmax: float
    vmin: float
    vmax: float
    advanced: SpectralAdvancedModel
