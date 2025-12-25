import numpy as np
from joblib import Parallel, delayed
from stockwell import st


from .models import Spectra, Seismogram, SpectralModel, SpectralAdvancedModel
from .utils import _apply_padding, _apply_smoothing, _get_wavenumbers_and_frequencies


HEADER_OFFSET_IND = 7



class SFK:
    def __init__(self, spec_params: SpectralModel):
        self._min_frequency = spec_params.fmin
        self._max_frequency = spec_params.fmax
        self._min_velocity = spec_params.vmin
        self._max_velocity = spec_params.vmax
        self._desired_nx = spec_params.advanced.desired_nx
        self._desired_nt = spec_params.advanced.desired_nt
        self._is_smooth_data = spec_params.advanced.smooth_data
        self._width = spec_params.advanced.width

    def __repr__(self):
        return "sfk"

    @staticmethod
    def _frequency_loop(tx_slice: np.ndarray, nt: int, nx: int, pad_nx: int, incline: np.ndarray) -> np.ndarray:
        fk_vg = np.zeros(shape=(incline.shape[1], pad_nx), dtype=np.complex64)
        fk_vg[:, :nx] = tx_slice[incline, np.arange(nx)[:, np.newaxis]].T

        fk_vg2 = np.fliplr(np.abs(np.fft.fft(fk_vg, axis=1)))
        fk2d_f_i = np.amax(fk_vg2, axis=0)

        return fk2d_f_i

    def _get_incline(self, nt: int, nx: int, dx: float, dt: float, offset_min: float) -> np.ndarray:
        nt_min = np.round(dx / self._max_velocity / dt)
        nt_max = np.min([np.round(dx / self._min_velocity / dt), nt])
        incline = np.clip(np.arange(nt_min, nt_max) * np.arange(nx)[:, np.newaxis], a_min=0, a_max=nt).astype(int)
        # incline = np.clip(np.arange(nt) * np.arange(nx)[:, np.newaxis], a_min=0, a_max=nt).astype(int)
        veloc = np.linspace(0, nx, nx) * dx / (incline + 1e-7).T
        t0 = (offset_min / (veloc.T + 1e-5)).astype(int)
        t0[0, :] = np.copy(t0[1, :])
        return np.clip(incline + t0, a_min=0, a_max=nt - 1).astype(int)

    def _st_tx2ftx(self, seismogram: Seismogram, ind_min_frequency: int, ind_max_frequency: int) -> np.ndarray:
        nt, nx = seismogram.time_counts, seismogram.spatial_counts
        print("data shape", seismogram.data.shape, ind_min_frequency, ind_max_frequency, nt, nx)
        ftx = np.zeros(shape=(ind_max_frequency - ind_min_frequency, nt, nx), dtype=np.complex64)
        # S-transform
        for idx in range(nx):
            ftx[:, :, idx] = st.st(
                np.float64(seismogram.data[:, idx]),
                ind_min_frequency,
                ind_max_frequency,
                self._width,
            )[1:, :]

        return ftx

    def _ftx2fk(self, seismogram: Seismogram) -> Spectra:
        nt, nx = seismogram.time_counts, seismogram.spatial_counts
        k, freq, ind_min_frequency, ind_max_frequency = _get_wavenumbers_and_frequencies(
            seismogram, self._min_frequency, self._max_frequency
        )

        k = np.fft.fftfreq(self._desired_nx, seismogram.dx)[: self._desired_nx // 2]
        # Stockwell transform compute
        ftx = self._st_tx2ftx(seismogram, ind_min_frequency, ind_max_frequency)

        offsets = seismogram.headers[HEADER_OFFSET_IND]
        offset_min = np.min(np.abs(offsets))

        incline = self._get_incline(nt, nx, seismogram.dx, seismogram.dt, offset_min)

        fk2d = np.array(
            Parallel(n_jobs=-1, backend='threading', prefer='threads')(
                delayed(SFK._frequency_loop)(ftx[freq_i], nt, nx, max(nx, self._desired_nx), incline)
                for freq_i in range(ftx.shape[0])
            )
        )

        return Spectra(
            vf_spectra=np.zeros((int(self._max_velocity - self._min_velocity), freq.size)),
            velocities=np.arange(self._min_velocity, self._max_velocity, 1),
            frequencies=freq,
            fk_spectra=fk2d,
            wave_numbers=k / 2,
        )

    def run(self, seismogram: Seismogram) -> Spectra:
        """
        Вычисление дисперсионного изображения.

        Вычисление дисперсионного изображения с помощью
        преобразования Стоквелла.

        Paramerets
        ----------
        seismogram: Seismogramm
            Сейсмические данные.

        Returns
        -------
        spectra: Spectra
            Дисперисонное изображение.
        """
        if self._is_smooth_data:
            seismogram = _apply_smoothing(seismogram)
        pad_seismogram = _apply_padding(seismogram, self._desired_nx, self._desired_nt, only_nt=True)
        spectra = self._ftx2fk(pad_seismogram)

        return spectra
