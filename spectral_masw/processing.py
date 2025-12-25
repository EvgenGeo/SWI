# spectral_masw/processing.py
import numpy as np
from .neural_net import NeuralNetworkManager
from .spectral_transform.ae_peaker import PeakerAE
from .spectral_transform.sfk import SFK, SpectralModel, SpectralAdvancedModel, HEADER_OFFSET_IND
from .spectral_transform.models import Seismogram, Spectra
from .spectral_transform.transformer import TransformerFK2VF


def process_seismogram(
    traces: np.ndarray,
    headers: np.ndarray,
    dt: float,
    data_type: str,
    f_min: float,
    f_max: float,
    v_min: float,
    v_max: float,
) -> dict:
    """
    ТВОЯ ФУНКЦИЯ СПЕКТРАЛЬНОГО АНАЛИЗА.

    Вход:
        traces      — матрица сейсмограммы (n_traces x n_samples)
        sample_rate — частота дискретизации
        f_min,f_max — мин/макс частоты (параметры пользователя)
        v_min,v_max — мин/макс скорости (параметры пользователя)
        model       — уже загруженная нейросеть (get_model(...))

    ДОЛЖЕН вернуть dict:
        'spec_image' : 2D np.ndarray (частоты x скорости)
        'freq_axis'  : 1D np.ndarray (частоты, размер = n_f)
        'vel_axis'   : 1D np.ndarray (скорости, размер = n_v)
        'curve_freq' : 1D np.ndarray (частоты кривой)
        'curve_vel'  : 1D np.ndarray (скорости кривой)
    """

    # ========= ЗАГЛУШКА / ШАБЛОН =========
    # Здесь ты полностью заменяешь логику на свою:
    # 1. считаешь спектральное изображение по MASW
    # 2. подаёшь его в model
    # 3. из результата модели извлекаешь кривую

    n_f, n_v = 128, 128
    spec_image = np.random.rand(n_f, n_v).astype('float32')
    freq_axis = np.linspace(f_min, f_max, n_f).astype('float32')
    vel_axis  = np.linspace(v_min, v_max, n_v).astype('float32')
    curve_freq = np.linspace(f_min, f_max, 50).astype('float32')
    curve_vel  = np.linspace(v_min, v_max, 50).astype('float32')

    if data_type == '2d':
        headers[HEADER_OFFSET_IND] = np.sqrt(
            (headers[1] - headers[3])**2 + (headers[2] - headers[4])**2
        )

    print('Done ofsets')

    seism = Seismogram(dx=np.int32(np.mean(np.diff(headers[HEADER_OFFSET_IND]))),
                       dt=dt/1e6,
                       data=traces.T,
                       headers=headers)
    #

    print('Done seism')
    spectral_adv = SpectralAdvancedModel(
        desired_nt=3000,
        desired_nx=500,
        width=1,
        smooth_data=True
    )
    spectral = SpectralModel(
        fmin=f_min,
        fmax=f_max,
        vmin=v_min,
        vmax=v_max,
        advanced=spectral_adv
    )

    spectra = SFK(spectral).run(seism)
    TransformerFK2VF.run(spectra)


    model = NeuralNetworkManager().get_model(model_name="AE_model_fundamental_mode")

    spec_image = spectra.vf_spectra
    freq_axis = spectra.frequencies
    vel_axis = spectra.velocities

    px, py = PeakerAE.peak_dc(spectra, model)

    image_size = spectra.vf_spectra.shape
    print(image_size)

    curve_vel = (px[0]) / (image_size[1]) * (f_max - f_min) + f_min
    curve_freq = (py[0]) / (image_size[0]) * (v_max - v_min) + v_min

    return {
        'spec_image': spec_image,
        'freq_axis': freq_axis,
        'vel_axis': vel_axis,
        'curve_freq': curve_freq,
        'curve_vel': curve_vel,
    }
