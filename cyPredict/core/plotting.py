"""Notebook-oriented Plotly chart helpers for cyPredict.

These helpers intentionally preserve the legacy notebook visual output. They
are not the future GammaSignalForge web plotting API; they are only the
notebook diagnostic layer used by the current research workflows.
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler


class PlottingMixin:
    """Render legacy notebook charts without affecting calculation paths."""

    def plot_single_range_analysis_charts(
        self,
        *,
        frequency_range,
        harmonics_amplitudes,
        original_data,
        data_column_name,
        index_of_max_time_for_cd,
    ):
        """Display the legacy single-range spectrum and reconstruction charts."""

        spectrum_trace = go.Scatter(
            x=frequency_range,
            y=harmonics_amplitudes,
            mode="lines",
            name="Goetzel DFT Spectrum",
        )
        fig_spectrum = go.Figure(spectrum_trace)
        fig_spectrum.update_layout(
            title="Frequency Spectrum",
            xaxis=dict(title="Frequency"),
            yaxis=dict(title="Magnitude"),
        )
        fig_spectrum.show()

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Original Data",
                "Detrended Data",
                "Dominant Circles Signal",
                "Centered Averages Delta",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=original_data.index,
                y=original_data[data_column_name],
                mode="lines",
                name="Original data",
            ),
            row=1,
            col=1,
        )

        fig.add_shape(
            type="line",
            x0=index_of_max_time_for_cd,
            x1=index_of_max_time_for_cd,
            y0=original_data[data_column_name].min(),
            y1=original_data[data_column_name].max(),
            line=dict(color="purple", width=1),
            row=1,
            col=1,
        )

        scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_detrended = scaler.fit_transform(
            original_data["detrended"].values.reshape(-1, 1)
        ).flatten()
        normalized_composite_circles = scaler.fit_transform(
            original_data["composite_dominant_circles_signal"].values.reshape(-1, 1)
        ).flatten()

        self.log_debug(
            "Chart data last valid indexes",
            function="analyze_and_plot",
            original_last_valid_index=original_data[data_column_name].last_valid_index(),
            detrended_last_valid_index=original_data["detrended"].last_valid_index(),
            cdc_last_valid_index=original_data[
                "composite_dominant_circles_signal"
            ].last_valid_index(),
        )

        fig.add_trace(
            go.Scatter(
                x=original_data.index,
                y=normalized_detrended,
                mode="lines",
                name="Detrended Close",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=original_data.index,
                y=normalized_composite_circles,
                mode="lines",
                name="Dominant Circle Signal",
            ),
            row=2,
            col=1,
        )

        fig.add_shape(
            type="line",
            x0=index_of_max_time_for_cd,
            x1=index_of_max_time_for_cd,
            y0=-1,
            y1=+1,
            line=dict(color="purple", width=1),
            row=2,
            col=1,
        )

        fig.update_layout(title="Goertzel Dominant Cyrcles Analysis", height=800)
        fig.update_xaxes(type="category")
        fig.show()

        return fig_spectrum, fig

    def plot_multiperiod_analysis_charts(
        self,
        *,
        reduced_data,
        data_column_name,
        composite_signal,
        elaborated_data_series,
        max_length_series_index,
        scaled_composite_signal,
        scaled_goertzel_composite_signal,
        scaled_detrended,
        scaled_alignmentsKPI,
        scaled_weigthed_alignmentsKPI,
        index_of_max_time_for_cd,
    ):
        """Display the legacy multirange notebook projection chart."""

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                "Original Data - " + self.ticker + " " + data_column_name + " Price",
                "Composite Domaninant Cycles Signal",
                "Cycles Alignment Indicators",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=reduced_data.index,
                y=reduced_data[data_column_name],
                mode="lines",
                name="Original data",
            ),
            row=1,
            col=1,
        )

        missing_values = composite_signal["composite_signal"].isnull().any()
        if missing_values:
            self.log_warning(
                "Missing values detected in composite signal",
                function="multiperiod_analysis",
            )

        x = elaborated_data_series[max_length_series_index].index

        fig.add_trace(
            go.Scatter(
                x=x,
                y=scaled_composite_signal,
                mode="lines",
                name="Composite Domaninant Cycles Signal GA Refactored",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=scaled_goertzel_composite_signal,
                mode="lines",
                name="Composite Domaninant Cycles Signal Goertzel Refactored",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=scaled_detrended,
                mode="lines",
                name="Detrended Signal (max lambda, minimal detrended)",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=scaled_alignmentsKPI,
                mode="lines",
                name="Cycles Alignment Indicator",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=scaled_weigthed_alignmentsKPI,
                mode="lines",
                name="Weigthed Cycles Alignment Indicator",
                yaxis="y2",
            ),
            row=3,
            col=1,
        )

        for row in (1, 2, 3):
            fig.add_vline(
                x=index_of_max_time_for_cd,
                line=dict(color="red", dash="dot"),
                name="Current Date",
                row=row,
                col=1,
            )

        samples_visible_before = 80
        samples_visible_after = 80
        index_center = index_of_max_time_for_cd
        start_range = max(0, index_center - samples_visible_before)
        end_range = index_center + samples_visible_after

        fig.update_xaxes(range=[start_range, end_range])
        fig.update_xaxes(type="category")

        fig.update_layout(
            title="Goertzel Dominant Cyrcles Analysis",
            height=1000,
            autosize=True,
            margin=dict(l=40, r=40, t=240, b=40),
            legend=dict(orientation="h", yanchor="top", y=1.2, xanchor="center", x=0.5),
        )
        fig.update_layout(yaxis=dict(autorange=True, fixedrange=False))

        y = scaled_composite_signal
        max_indices = argrelextrema(y, np.greater)[0]
        min_indices = argrelextrema(y, np.less)[0]
        max_datetimes = x[max_indices]
        min_datetimes = x[min_indices]

        def pick_extrema_near_target(datetimes, target_index, count_after=2, count_before=1):
            target_dt = x[target_index]
            dt_series = pd.Series(datetimes)
            after = dt_series[dt_series > target_dt].sort_values().head(count_after)
            before = dt_series[dt_series <= target_dt].sort_values(ascending=False).head(count_before)
            return before.tolist() + after.tolist()

        relevant_max_dt = pick_extrema_near_target(max_datetimes, index_of_max_time_for_cd)
        relevant_min_dt = pick_extrema_near_target(min_datetimes, index_of_max_time_for_cd)

        is_intraday = pd.Series(reduced_data.index).diff().mode()[0] < pd.Timedelta("1D")

        used_points = []
        for dt in relevant_max_dt:
            y_val = scaled_composite_signal[x.get_loc(dt)]
            count_same_y = sum(
                1
                for xd, yd in used_points
                if abs(yd - y_val) < 0.01 and abs((dt - xd).total_seconds()) < 60 * 60
            )
            offset_y = 0.02 + count_same_y * 0.05
            ay_val = -30 - count_same_y * 15
            used_points.append((dt, y_val))

            fig.add_annotation(
                x=dt,
                y=y_val + offset_y,
                text=dt.strftime("%H:%M") if is_intraday else dt.strftime("%Y-%m-%d"),
                showarrow=True,
                arrowhead=2,
                arrowside="end",
                arrowcolor="red",
                arrowwidth=2.5,
                ax=0,
                ay=ay_val,
                row=2,
                col=1,
            )

        used_points = []
        for dt in relevant_min_dt:
            y_val = scaled_composite_signal[x.get_loc(dt)]
            count_same_y = sum(
                1
                for xd, yd in used_points
                if abs(yd - y_val) < 0.01 and abs((dt - xd).total_seconds()) < 60 * 60
            )
            offset_y = 0.02 + count_same_y * 0.05
            ay_val = 30 + count_same_y * 15
            used_points.append((dt, y_val))

            fig.add_annotation(
                x=dt,
                y=y_val - offset_y,
                text=dt.strftime("%H:%M") if is_intraday else dt.strftime("%Y-%m-%d"),
                showarrow=True,
                arrowhead=2,
                arrowside="end",
                arrowcolor="green",
                arrowwidth=2.5,
                ax=0,
                ay=ay_val,
                row=2,
                col=1,
            )

        x_range_start = reduced_data.index[start_range]
        x_range_end = reduced_data.index[min(end_range, len(reduced_data.index) - 1)]
        idx_start = reduced_data.index.searchsorted(x_range_start)
        idx_end = reduced_data.index.searchsorted(x_range_end)

        visible_y = reduced_data.iloc[idx_start:idx_end][data_column_name].dropna()
        if not visible_y.empty:
            ymin = visible_y.min()
            ymax = visible_y.max()
            fig.update_yaxes(
                range=[ymin - 10, ymax + 10],
                autorange=False,
                fixedrange=False,
                row=1,
                col=1,
            )

        visible_x_range = x[idx_start:idx_end]
        visible_cdc = pd.Series(scaled_composite_signal, index=x).loc[visible_x_range].dropna()
        if not visible_cdc.empty:
            ymin2 = visible_cdc.min()
            ymax2 = visible_cdc.max()
            fig.update_yaxes(
                range=[ymin2 - 20, ymax2 + 20],
                autorange=False,
                fixedrange=False,
                row=2,
                col=1,
            )

        all_extremes = sorted(relevant_min_dt + relevant_max_dt)
        for i in range(len(all_extremes) - 1):
            t0 = all_extremes[i]
            t1 = all_extremes[i + 1]
            color = "rgba(255, 0, 0, 0.2)" if t0 in relevant_max_dt else "rgba(0, 255, 0, 0.2)"
            for row in (1, 2, 3):
                fig.add_vrect(
                    x0=t0,
                    x1=t1,
                    fillcolor=color,
                    opacity=0.2,
                    line_width=0,
                    row=row,
                    col=1,
                )

        x_list = list(x)
        all_tagged_extremes = sorted(relevant_min_dt + relevant_max_dt)
        past_extremes = [dt for dt in all_tagged_extremes if dt <= x[index_of_max_time_for_cd]]
        future_extremes = [dt for dt in all_tagged_extremes if dt > x[index_of_max_time_for_cd]]

        for i, dt in enumerate(past_extremes + future_extremes):
            center_idx = x_list.index(dt)
            delta = 3 if dt in past_extremes or i == len(past_extremes) else 4
            start_idx = max(0, center_idx - delta)
            end_idx = min(len(x_list) - 1, center_idx + delta)
            t0 = x_list[start_idx]
            t1 = x_list[end_idx]
            for row in (1, 2, 3):
                fig.add_vrect(
                    x0=t0,
                    x1=t1,
                    fillcolor="rgba(150,150,150,0.60)",
                    opacity=0.25,
                    line_width=0,
                    row=row,
                    col=1,
                )

        fig.show()
        plot(fig, filename=f"multirange analysis for {self.ticker}.html", auto_open=False)

        self.log_debug(
            "Chart index diagnostics",
            function="multiperiod_analysis",
            reduced_index_type=type(reduced_data.index),
            reduced_index_timezone=reduced_data.index.tz,
            elaborated_index_type=type(x),
            elaborated_index_timezone=x.tz,
        )

        return fig
