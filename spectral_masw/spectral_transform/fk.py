import numpy as np

from src.config_reader.models import SpectralModel
from src.spectral_analysis.models import Seismogram, Spectra
from src.spectral_analysis.utils import _apply_padding, _apply_smoothing, _get_wavenumbers_and_frequencies


class FKT:
    def __init__(self, spec_params: SpectralModel) -> None:
        self._min_frequency = spec_params.fmin
        self._max_frequency = spec_params.fmax
        self._min_velocity = spec_params.vmin
        self._max_velocity = spec_params.vmax
        self._desired_nx = spec_params.advanced.desired_nx
        self._desired_nt = spec_params.advanced.desired_nt
        self._is_smooth_data = spec_params.advanced.smooth_data

    def __repr__(self):
        return "fk"

    def _tx2fk(self, seismogram: Seismogram) -> Spectra:
        fk2d = np.fliplr(np.abs(np.fft.fft2(seismogram.data)))

        k, freq, ind_min_frequency, ind_max_frequency = _get_wavenumbers_and_frequencies(
            seismogram, self._min_frequency, self._max_frequency
        )

        return Spectra(
            vf_spectra=np.zeros((
                int(self._max_velocity - self._min_velocity),
                ind_max_frequency - ind_min_frequency,
            )),
            velocities=np.arange(self._min_velocity, self._max_velocity, 1),
            frequencies=freq,
            fk_spectra=fk2d[ind_min_frequency:ind_max_frequency, :],
            wave_numbers=k / 2,
        )

    def run(self, seismogram: Seismogram) -> Spectra:
        """
        Вычисление дисперсионного изображения.

        Вычисление дисперсионного изображения с помощью двумерного
        преобразования Фурье.

        Parameters
        ----------
        seismogram: Seismogram
            Сейсмические данные.

        Returns
        -------
        spectra: Spectra
            Дисперсионное изображение.
        """

        if self._is_smooth_data:
            seismogram = _apply_smoothing(seismogram)

        pad_seismogram = _apply_padding(seismogram, self._desired_nx, self._desired_nt, only_nt=False)

        spectra = self._tx2fk(pad_seismogram)

        return spectra
