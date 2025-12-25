from typing import Optional
import numpy as np
import hdbscan
import cv2
from scipy.interpolate import interp1d

# from .base_peaker import Peaker
from .models import Spectra

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
    import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

np.random.seed(42)
SHAPE = 256


class PeakerAE:
    """Picking dispersion curves using a neural network."""
    @staticmethod
    def _segments_fit(x_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_uniq = np.unique(x_test)
        y_uniq = []
        for x in x_uniq:
            condition = x_test == x
            y_uniq.append(np.mean(y_test[condition]))
        y_uniq = np.asarray(y_uniq)
        return x_uniq, y_uniq

    @staticmethod
    def _get_coeffs(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Calculate linear regression coefficients for given x and y data.

        Args:
            x: Independent variable values (frequency indices)
            y: Dependent variable values (velocity values)

        Returns:
            tuple: Intercept (b) and slope (a) coefficients of linear fit y = a*x + b
        """
        A = np.array([np.ones_like(x), x]).T
        A_inv = np.linalg.pinv(A)
        coeffs = A_inv @ y
        return coeffs[0], coeffs[1]

    @staticmethod
    def _remove_outbreaks(data: np.ndarray, fcount: int, vcount: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from data using HDBSCAN clustering.

        Identifies the largest cluster and returns only points belonging to that cluster,
        effectively removing outlier points.

        Args:
            data: 2D array of data points to filter (shape: n_samples, 2)

        Returns:
            tuple: Filtered x and y coordinates of the largest cluster
        """
        labels = hdbscan.HDBSCAN(core_dist_n_jobs=1).fit_predict(data)

        uniq_labels = np.unique(labels)
        valid_labels = uniq_labels[uniq_labels>=0]
        if len(valid_labels):
            all_clusters = [data[labels == i] for i in valid_labels]

            distance = np.zeros(len(valid_labels))
            xrange = np.zeros(len(valid_labels))
            size = np.zeros(len(valid_labels))
            for ind, tmp_cluster in enumerate(all_clusters):
                distance[ind] = np.mean(tmp_cluster[:, 1] / fcount) ** 2 + np.mean(tmp_cluster[:, 0] / vcount) ** 2
                xrange[ind] = len(np.unique(tmp_cluster[:, 0]))
                size[ind] = len(tmp_cluster)

            distance /= np.max(distance)
            distance = 1-distance
            distance /= np.max(distance)
            xrange /= np.max(xrange)
            size /= np.max(size)

            metrics = distance + 0.5*xrange + 0.25*size
            best_cluster = np.argmax(metrics)
            x = all_clusters[best_cluster][:, 0]
            y = all_clusters[best_cluster][:, 1]
        else:
            x = data[:, 0]
            y = data[:, 1]

        return x, y

    @staticmethod
    def _get_curve(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a dispersion curve from spectral data.

        Processes input data to identify the dominant curve by:
        1. Thresholding to keep only high-amplitude points (top 5%)
        2. Clustering to remove outliers
        3. Linear fitting and statistical outlier removal (3σ criterion)
        4. Segment averaging

        Args:
            data: 2D spectral data array (velocity-frequency matrix)

        Returns:
            tuple: Processed x (frequency indices) and y (velocity indices) coordinates of the curve
        """
        y_size = data.shape[0]
        x_size = data.shape[1]
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        xx, yy = np.meshgrid(x, y)
        condition = data >= np.max(data)*0.95
        x_ = xx[condition]
        y_ = yy[condition]

        if len(x_) == 1:
            return np.array([]), np.array([])

        data_for_clust = np.array([x_, y_]).T
        # indexes = data_for_clust[:, 1] > round(y_size/SHAPE)
        data_for_clust_new = np.copy(data_for_clust)
        # data_for_clust_new[:, 1][indexes] = data_for_clust_new[:, 1][indexes] - round(y_size/SHAPE)
        x_, y_ = PeakerAE._remove_outbreaks(data_for_clust_new, x_size, y_size)
        b, a = PeakerAE._get_coeffs(x_, y_)
        std = 3 * np.std(y_ - (a * x_ + b))

        trend = a * x_ + b
        condition = ((trend - 3 * std) < y_) * ((trend + 3 * std) > y_)

        x__ = x_[condition]
        y__ = y_[condition]

        px, py = PeakerAE._segments_fit(x__, y__)

        return px, py

    @staticmethod
    def _image_resize(img, target_size=(SHAPE, SHAPE)):
        reshaped_img = cv2.resize(
            img.astype(np.float32),
            target_size,
            interpolation=cv2.INTER_AREA
        )
        return reshaped_img

    @staticmethod
    def predict_mask(vf_spectra: np.ndarray, model) -> np.ndarray:

        main_shape = vf_spectra.shape
        vf_spectra4predict = PeakerAE._image_resize(vf_spectra, target_size=(SHAPE, SHAPE))

        input_tensor = tf.convert_to_tensor(
            vf_spectra4predict.reshape(1, SHAPE, SHAPE, 1),
            dtype=tf.float32
        )

        mask = model(input_tensor, training=False).numpy()

        return PeakerAE._image_resize(mask[0, :, :, 0], target_size=(main_shape[1], main_shape[0]))

    @staticmethod
    def _update_curves_mas(mask, px, py):
        px_tmp, py_tmp = PeakerAE._get_curve(mask)
        sort_indexes = np.argsort(px_tmp)
        px.append(px_tmp[sort_indexes])
        py.append(py_tmp[sort_indexes])
        return px, py

    @staticmethod
    def peak_dc(spectra: Spectra, model) \
            -> tuple[list, list]:

        mask = PeakerAE.predict_mask(spectra.vf_spectra, model)
        px, py = [], []
        px, py = PeakerAE._update_curves_mas(mask, px, py)

        return px, py