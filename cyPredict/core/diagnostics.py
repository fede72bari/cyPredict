"""Diagnostic helpers for the legacy cycle-analysis engine."""


class DiagnosticsMixin:
    """Expose lightweight diagnostics retained for notebook workflows."""

    def debug_check_complex_col(self, colname):
        """Log rows with complex values in ``self.MultiAn_dominant_cycles_df``.

        Parameters
        ----------
        colname : str
            Column to inspect. Missing columns are ignored, preserving the
            historical non-failing behavior.
        """
        if colname not in self.MultiAn_dominant_cycles_df.columns:
            return

        df = self.MultiAn_dominant_cycles_df
        complex_mask = df[colname].apply(lambda val: isinstance(val, complex))
        if complex_mask.any():
            indices = df.index[complex_mask]
            for i in indices:
                val = df.at[i, colname]
                self.log_debug(
                    "Complex value detected in dominant-cycle dataframe",
                    function="debug_check_complex_col",
                    index=i,
                    column=colname,
                    value=val,
                )

    def debug_check_complex_values(self):
        """Inspect amplitude, frequency and phase columns for complex values."""
        cols = ["peak_amplitudes", "peak_frequencies", "peak_phases"]
        for c in cols:
            self.debug_check_complex_col(c)

    def get_goertzel_amplitudes(self):
        """Return the last Goertzel amplitude list stored on the instance."""
        return self.goertzel_amplitudes
