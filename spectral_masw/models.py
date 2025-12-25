from django.db import models
import numpy as np
import pickle

class SegyFile(models.Model):
    name = models.CharField(max_length=255)
    original_file = models.FileField(upload_to='segy/')
    upload_date = models.DateTimeField(auto_now_add=True)

    sample_rate = models.FloatField()     # Гц или мкс — как у тебя
    num_samples = models.IntegerField()
    num_traces = models.IntegerField()

    # Матрица трасс (n_traces x n_samples), хранится как pickle numpy float32
    trace_data = models.BinaryField()

    def __str__(self):
        return self.name


class TraceHeader(models.Model):
    """Заголовки трасс с ключевыми параметрами сейсморазведки"""
    segy_file = models.ForeignKey(
        SegyFile,
        on_delete=models.CASCADE,
        related_name='traces',
        verbose_name="SEG-Y файл"
    )

    # Основные идентификаторы
    trace_number = models.IntegerField(verbose_name="Номер трассы в файле")
    ffig = models.IntegerField(verbose_name="FFIG - Field File Identification Number")

    # Координаты приёмников (RECX/Y)
    rec_x = models.FloatField(verbose_name="RECX - X координата приёмника")
    rec_y = models.FloatField(verbose_name="RECY - Y координата приёмника")

    # Координаты источников (SOUX/Y)
    sou_x = models.FloatField(verbose_name="SOUX - X координата источника")
    sou_y = models.FloatField(verbose_name="SOUY - Y координата источника")

    # Расстояние и высота
    offset = models.FloatField(verbose_name="Offset - расстояние приём-источник")
    elevation = models.FloatField(verbose_name="Elevation - высота")

    # CDP информация
    cdp_x = models.FloatField(verbose_name="CDPX - X координата CDP")
    cdp_y = models.FloatField(verbose_name="CDPY - Y координата CDP")
    cdp = models.IntegerField(verbose_name="CDP - Common Depth Point")

    # Дополнительные параметры трассы
    dt = models.IntegerField(verbose_name="DT - интервал дискретизации (микросек)")

    class Meta:
        db_table = 'spectral_trace_headers'
        verbose_name = "Заголовок трассы"
        verbose_name_plural = "Заголовки трасс"
        indexes = [
            models.Index(fields=['segy_file', 'trace_number']),
            models.Index(fields=['cdp']),
            models.Index(fields=['offset']),
        ]

    def __str__(self):
        return f'{self.segy_file.name} #{self.trace_number}'


class SpectralResult(models.Model):
    """
    Результат спектрального анализа одной сейсмограммы.
    Хранит:
      - спектральное изображение (частота x скорость)
      - оси частот и скоростей (как периоды extent)
      - кривую (частоты/скорости)
    """
    segy_file = models.ForeignKey(SegyFile, on_delete=models.CASCADE, related_name='spectra')
    name = models.CharField(max_length=255, default='Результат спектрального анализа')

    created_at = models.DateTimeField(auto_now_add=True)

    # 1) Спектральное изображение — матрица float32 (freq x vel)
    spectrogram_matrix = models.BinaryField()

    # 2) Пределы осей (скорость — X, частота — Y)
    extent_x_min = models.FloatField()   # min скорость
    extent_x_max = models.FloatField()   # max скорость
    extent_y_min = models.FloatField()   # min частота
    extent_y_max = models.FloatField()   # max частота

    # 3) Кривая: массив скоростей и частот (в физических единицах)
    curve_x_array = models.BinaryField()   # скорости кривой
    curve_y_array = models.BinaryField()   # частоты кривой
    processing_notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.name} ({self.segy_file.name})'

    # Утилиты
    def set_spectrogram(self, mat: np.ndarray):
        self.spectrogram_matrix = pickle.dumps(mat.astype('float32'))

    def get_spectrogram(self) -> np.ndarray:
        return pickle.loads(self.spectrogram_matrix)

    def set_curve_arrays(self, v: np.ndarray, f: np.ndarray):
        self.curve_x_array = pickle.dumps(v.astype('float32'))
        self.curve_y_array = pickle.dumps(f.astype('float32'))

    def get_curve_x(self) -> np.ndarray:
        return pickle.loads(self.curve_x_array)

    def get_curve_y(self) -> np.ndarray:
        return pickle.loads(self.curve_y_array)
