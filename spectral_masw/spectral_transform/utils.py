import numpy as np
from .models import Seismogram



def _apply_smoothing(seismogram: Seismogram) -> Seismogram:
    """
   Applies a 2D smoothing window to the seismic data.

   This function applies a 2D smoothing window to the seismic data in the
   input `Seismogram` object. The window is constructed by taking the outer product
   of two 1D filters, one for the time dimension and one for the spatial
   dimension. The size of the smoothing window is determined by a fraction
   (1/20) of the number of time samples and spatial samples.

   Args:
       seismogram (Seismogram): A Seismogram object containing seismic data.

   Returns:
       Seismogram: A new Seismogram object with the smoothed seismic data.
   """
    nt, nx = seismogram.time_counts, seismogram.spatial_counts
    length_smooth = map(lambda size: (int(size / 20), size), (nt, nx))
    filter_time, filter_space = [
        np.pad(
            np.ones(size - 2 * length_size),
            pad_width=(length_size, length_size),
            mode="linear_ramp",
        )
        for length_size, size in length_smooth
    ]
    window_2d = np.outer(filter_time, filter_space)
    return Seismogram(window_2d * seismogram.data, seismogram.headers, seismogram.dt, seismogram.dx)


def _apply_padding(seismogram: Seismogram, desired_nx: int, desired_nt: int, only_nt: bool = True) -> Seismogram:
    """
    Applies padding to the seismic data.

    This function applies padding to the seismic data in the input `Seismogram`
    object. It pads the data to achieve the desired number of time samples
    (`desired_nt`) and/or spatial samples (`desired_nx`). Padding is applied
    only in the time dimension if `only_nt` is True; otherwise, padding is
    applied in both time and spatial dimensions.

    Args:
        seismogram (Seismogram): A Seismogram object containing seismic data.
        desired_nx (int): The desired number of spatial samples after padding.
        desired_nt (int): The desired number of time samples after padding.
        only_nt (bool, optional): A flag indicating whether to pad only in the
            time dimension (True) or in both time and spatial dimensions (False).
            Defaults to True.

    Returns:
        Seismogram: A new Seismogram object with the padded seismic data.
    """
    nt, nx = seismogram.time_counts, seismogram.spatial_counts
    new_nx, new_nt = max(nx, desired_nx), max(nt, desired_nt)
    pad_width = ((0, new_nt - nt), (0, 0) if only_nt else (0, new_nx - nx))
    pad_data = np.pad(seismogram.data, pad_width=pad_width)

    return Seismogram(pad_data, seismogram.headers, seismogram.dt, seismogram.dx)


def _get_wavenumbers_and_frequencies(
    seismogram: Seismogram, min_frequency: float, max_frequency: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Calculates wavenumbers and frequencies for spectral analysis.

    This function calculates the wavenumbers (k) and frequencies (freq) based on
    the properties of the input seismogram and the specified minimum and maximum
    frequencies. It also determines the indices corresponding to the minimum and
    maximum frequencies for use in later processing steps.

    Args:
        seismogram (Seismogram): A Seismogram object containing seismic data
            and related parameters (dt, time_counts, dx, spatial_counts).
        min_frequency (float): The minimum frequency for the analysis.
        max_frequency (float): The maximum frequency for the analysis.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, int]: A tuple containing:
            - k (np.ndarray): NumPy array of wavenumbers.
            - freq (np.ndarray): NumPy array of frequencies between min_frequency and max_frequency.
            - ind_min_frequency (int): Index corresponding to the minimum frequency.
            - ind_max_frequency (int): Index corresponding to the maximum frequency.
    """
    df = 1 / seismogram.dt / seismogram.time_counts
    dk = 1 / seismogram.dx / seismogram.spatial_counts
    print(df, dk, seismogram.dt, seismogram.dx, seismogram.time_counts, seismogram.spatial_counts)

    ind_min_frequency = int(np.round(min_frequency / df))
    ind_max_frequency = int(np.round(max_frequency / df))

    k = np.arange(0, seismogram.spatial_counts * dk, dk)
    freq = np.linspace(min_frequency, max_frequency, ind_max_frequency - ind_min_frequency)

    return k, freq, ind_min_frequency, ind_max_frequency

