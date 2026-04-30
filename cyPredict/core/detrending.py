"""Detrending filters used by the legacy cycle-analysis engine."""

import numpy as np
import pandas as pd
import scipy
from statsmodels.nonparametric.smoothers_lowess import lowess


class DetrendingMixin:
    """Apply the detrending filters selected by ``analyze_and_plot``."""

    def hp_filter(self, data, lambda_, ret=False):
        """Apply the Hodrick-Prescott filter implementation used by cyPredict.

        Parameters
        ----------
        data : pandas.Series
            Input series to filter.
        lambda_ : float
            HP smoothing parameter. Larger values produce smoother trends.
        ret : bool, default False
            When false, return the detrended component ``data - trend``.
            When true, return the trend component itself.

        Returns
        -------
        tuple
            ``(output, status_code)``. ``status_code`` is ``1`` on success,
            ``0`` when the input is too short, and ``3`` when the linear system
            hits a division-by-zero condition.
        """
        nobs = len(data)

        output = data.to_numpy(dtype=np.float64).copy()

        if nobs <= 5:
            self.log_warning("HP filter input too short", function="hp_filter", nobs=nobs)
            return None, 0  # Not enough data

        a = np.zeros(nobs)
        b = np.zeros(nobs)
        c = np.zeros(nobs)

        a[0] = 1.0 + lambda_
        b[0] = -2.0 * lambda_
        c[0] = lambda_

        for i in range(1, nobs - 2):
            a[i] = 6.0 * lambda_ + 1
            b[i] = -4.0 * lambda_
            c[i] = lambda_

        a[1] = 5.0 * lambda_ + 1
        a[nobs - 1] = 1.0 + lambda_
        a[nobs - 2] = 5.0 * lambda_ + 1.0

        b[nobs - 2] = -2.0 * lambda_

        H1 = 0
        H2 = 0
        H3 = 0
        H4 = 0
        H5 = 0
        HH1 = 0
        HH2 = 0
        HH3 = 0
        HH4 = 0
        HH5 = 0
        Z = 0
        HB = 0
        HC = 0

        for i in range(nobs):
            Z = a[i] - H4 * H1 - HH5 * HH2
            if Z == 0:
                self.log_debug(
                    "HP filter division by zero",
                    function="hp_filter",
                    index=i,
                    Z=Z,
                    a=a[i],
                    b=b[i],
                    c=c[i],
                    H1=H1,
                    H2=H2,
                    H3=H3,
                )

                return None, 3  # Division by zero
            HB = b[i]
            HH1 = H1
            H1 = (HB - H4 * H2) / Z
            b[i] = H1
            HC = c[i]
            HH2 = H2
            H2 = HC / Z
            c[i] = H2
            a[i] = (output[i] - HH3 * HH5 - H3 * H4) / Z
            HH3 = H3
            H3 = a[i]
            H4 = HB - H5 * HH1
            HH5 = H5
            H5 = HC

        H2 = 0
        H1 = a[nobs - 1]
        output[nobs - 1] = H1

        for i in range(nobs - 1, 0, -1):
            output[i - 1] = a[i - 1] - b[i - 1] * H1 - c[i - 1] * H2
            H2 = H1
            H1 = output[i - 1]

        if not ret:
            output = data - output

        return output, 1

    def jh_filter(self, y, p=4, h=8):
        """Apply the autoregressive JH-style detrending helper.

        Parameters
        ----------
        y : array-like
            Input signal.
        p : int, default 4
            Number of lagged values used as regression predictors.
        h : int, default 8
            Forecast horizon offset used to build the regression target.

        Returns
        -------
        numpy.ndarray
            Residual series produced by subtracting the fitted cyclical
            component from ``y[h:]``.
        """
        from sklearn.linear_model import LinearRegression

        n = len(y)

        self.log_debug("JH filter parameters", function="jh_filter", p=p, h=h)

        X = np.ones((n - h, p + 1))
        y_est = np.zeros(n - h)

        for i in range(p):
            X[:, i+1] = y[i:n-h+i]
        for i in range(n - h):
            y_est[i] = y[i+h]

        model = LinearRegression()
        model.fit(X, y_est)

        cyclical_component = model.predict(X)

        detrended_y = y[h:] - cyclical_component

        return detrended_y

    def linear_detrend(self, data, window_size=0):
        """Apply scipy linear detrending over complete fixed-size windows.

        Parameters
        ----------
        data : pandas.Series
            Input signal with an index that should be preserved.
        window_size : int, default 0
            Window length used to build breakpoints. Existing callers should
            pass a positive value for linear detrending; the legacy default is
            preserved for API compatibility.

        Returns
        -------
        pandas.Series
            Detrended data aligned to the original index.
        """
        if window_size == 0:
            break_points = 0
        else:
            total_length = len(data)
            remainder = total_length % window_size
            start_index = remainder

            num_complete_windows = (total_length - remainder) // window_size
            break_points = [start_index + i * window_size for i in range(1, num_complete_windows)]

            break_points = [bp for bp in break_points if bp < total_length]

        detrended_data = scipy.signal.detrend(data[start_index:], type='linear', bp=[bp - start_index for bp in break_points])

        detrended_data_full = np.concatenate((data[:start_index], detrended_data))

        return pd.Series(detrended_data_full, index=data.index)

    def detrend_lowess(self, signal, P_max, k=2):
        """Remove trend with LOWESS using a window derived from max period.

        Parameters
        ----------
        signal : array-like
            Signal to detrend.
        P_max : float
            Maximum period to preserve. The LOWESS window is ``k * P_max``.
        k : float, default 2
            Multiplicative factor used to derive the LOWESS window.

        Returns
        -------
        tuple
            ``(trend, residual)`` where ``residual`` is ``signal - trend``.
        """
        time = np.arange(len(signal))
        window = int(k * P_max)
        frac = window / len(time)

        trend = lowess(signal, time, frac=frac, return_sorted=False)

        residual = signal - trend

        return trend, residual
