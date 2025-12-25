# """
# Модуль для пикировная дисперсионных кривых.
#
# Модуль выполняет пикирование дисперсионных кривых по
# массиву энергии в зависимости от скорости и частоты в некоторых пределах.
# """
#
# import numpy as np
# from scipy.interpolate import interp1d
# from typing import Optional
# from typing_extensions import Self
# from pathlib import Path
#
# # from src.spectral_analysis.utils import get_dc_ranges_from_file
# from src.spectral_analysis.models import Spectra
# from sklearn.mixture import GaussianMixture
#
#
# class Peaker:
#     """Базовый класс для пикировки дисперсионных кривых."""
#
#     def __init__(
#         self,
#         path4dc_limits: Path
#     ) -> None:
#         """
#         Sets peaking ranges.
#
#         Args:
#             path4dc_limits (Path):
#                 Absolute path to the file with the reference dispersion curve and
#                 increment to determine the ranges for the frequency set.
#         """
#         self.path4dc_limits = path4dc_limits
#         self._interpolator = None
#         self._step_interpolator = None
#         self.f_p: Optional[np.ndarray]= None
#         self.v_min: Optional[np.ndarray] = None
#         self.v_max: Optional[np.ndarray] = None
#         self._v_min_interpolator = None
#         self._v_max_interpolator = None
#
#     @classmethod
#     def initialize(cls, path4dc_limits: Path) -> Self:
#         """
#         Initializes the Peaker class.
#
#         Reads in the frequency ranges and interpolation to determine pick values.
#
#         Args:
#             path4dc_limits (Path):
#                 Absolute path to the file with the reference dispersion curve and
#                 increment to determine the ranges for the frequency set.
#
#         Returns:
#             instance (Peaker): The object, with initialized interpolator.
#         """
#         instance = cls(path4dc_limits)
#
#         #чтение частот и диапазонов пикирования из файла
#         instance.f_p, instance.v_min, instance.v_max = get_dc_ranges_from_file(path4dc_limits)
#
#         # инициализация интерполятора для диапазонов пикировки
#         instance._v_min_interpolator = interp1d(instance.f_p, instance.v_min)
#         instance._v_max_interpolator = interp1d(instance.f_p, instance.v_max)
#
#         return instance
#
#     def _find_limits(self, spectra: Spectra) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Finds limits for interpolation
#
#         Select frequencies and velocities relative to the boundaries.
#
#         Args:
#             spectra (Spectra):
#                 Object contain the frequencies, velocities and spectrum.
#
#         Returns:
#              f_p_interp (np.ndarray): Interpolated frequencies.
#              upper_limits (np.ndarray): Upper limits for interpolation.
#              lower_limits (np.ndarray): Lower limits for interpolation.
#              f_indx_spectrum (np.ndarray): The spectrum indexes.
#         """
#         # выбор частот и скоростей относительно границ
#         f_p_interp = spectra.frequencies[
#             (self.f_p.min() <= spectra.frequencies) & (spectra.frequencies <= self.f_p.max())
#         ]
#
#         # Интерполяция диапазонов пикировки и проверка на выходны из диапазона возможных скоростей
#         v_min = self._v_min_interpolator(f_p_interp)
#         v_max = self._v_max_interpolator(f_p_interp)
#
#         lower_limits = np.ceil(np.clip(v_min, spectra.velocities[0], spectra.velocities[-1]))
#         upper_limits = np.ceil(np.clip(v_max, spectra.velocities[0], spectra.velocities[-1]))
#         lower_limits[lower_limits==upper_limits] = lower_limits[lower_limits==upper_limits]-1
#
#         _, _, f_indx_spectrum = np.intersect1d(f_p_interp, spectra.frequencies, return_indices=True)
#
#         return f_p_interp, upper_limits, lower_limits, f_indx_spectrum
#
#     def apply_croping(self, spectra: Spectra, peak_fraction: float):
#         """
#         Apply Cropping based on find limits
#
#         Crops based on find limits to mask the spectrum to apply.
#
#         Args:
#             spectra (Spectra):
#                 Object contain the frequencies, velocities and spectrum.
#             peak_fraction (float):
#                 peak fraction for spectral cropping.
#
#         Returns:
#              freq (np.ndarray): The frequencies.
#              upper_limits (np.ndarray): Upper limits for interpolation.
#              lower_limits (np.ndarray): Lower limits for interpolation.
#              f_indx_spectrum (np.ndarray): The spectrum indexes.
#              mask (np.ndarray): cropped Spectra mask
#         """
#         freq, upper_limits, lower_limits, f_indx_spectrum = self._find_limits(spectra)
#
#         shift = f_indx_spectrum[0]
#         min_vel = spectra.velocities[0]
#         mask = np.zeros_like(spectra.vf_spectra, dtype=bool)
#
#
#         for idx in range(len(freq)):
#             spectrum_slice = spectra.vf_spectra[int(lower_limits[idx]-min_vel) : int(upper_limits[idx]-min_vel), idx + shift]
#             threshold = np.max(spectrum_slice) * peak_fraction
#             mask[int(lower_limits[idx]-min_vel) : int(upper_limits[idx]-min_vel), idx + shift] = spectrum_slice >= threshold
#
#         return freq, upper_limits, lower_limits, f_indx_spectrum, mask
#
#     def peak_dc(self, spectra: Spectra, peak_fraction: float, cutoff_fraction: float) -> tuple[list, list, np.ndarray, np.ndarray]:
#         """
#         Extracts dispersion curves from the spectra.
#
#         Takes a spectra object and peaks the DC components to be returns.
#         This is a base function and should be overwritten.
#
#         Args:
#             spectra (Spectra):
#                 Object contain the frequencies, velocities and spectrum.
#             peak_fraction (float):
#                 peak fraction for spectral cropping.
#             cutoff_fraction (float):
#                 cutoff fraction for spectral cropping.
#
#         Returns:
#              list: This should be overwritten by the inherited class
#         """
#         pass
