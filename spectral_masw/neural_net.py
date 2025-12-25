# spectral_masw/neural_net.py
from pathlib import Path
from django.conf import settings
import os
import sys
from contextlib import contextmanager
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

@contextmanager
def suppress_stderr():
    fd = sys.stderr.fileno()
    def _redirect_stderr(to):
        sys.stderr.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w')  # Python function to write to fd

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(os.devnull, 'w') as file:
            _redirect_stderr(to=file)
        try:
            yield  # allow code to be run with the redirected stderr
        finally:
            _redirect_stderr(to=old_stderr)  # restore stderr

# Использование:
with suppress_stderr():
    from tensorflow.keras.models import load_model  # tf.keras


class NeuralNetworkManager:
    """
    Менеджер для загрузки и кеширования Keras-моделей (.h5).
    """

    MODELS_DIR = Path(settings.BASE_DIR) / 'neural_models'
    _cache: dict[str, object] = {}

    @classmethod
    def _ensure_dir(cls):
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model(cls, model_name: str = 'masw_model') -> object:
        """
        Возвращает загруженную Keras‑модель.
        Файл должен лежать как BASE_DIR/neural_models/<model_name>.h5
        """
        cls._ensure_dir()

        if model_name in cls._cache:
            return cls._cache[model_name]

        path = cls.MODELS_DIR / f'{model_name}.h5'
        if not path.exists():
            raise FileNotFoundError(
                f'Файл модели {path} не найден. '
                f'Положи свой .h5 в папку neural_models/'
            )

        model = load_model(path)  # Keras/TensorFlow загрузка [web:120][web:123]
        cls._cache[model_name] = model
        return model

    @classmethod
    def list_available_models(cls) -> list[str]:
        cls._ensure_dir()
        return [p.stem for p in cls.MODELS_DIR.glob('*.h5')]
