"""Spectral scoring helpers for the legacy cycle-analysis engine."""

import math


class SpectralMixin:
    """Small spectral-analysis helpers used by ``analyze_and_plot``."""

    def get_bartels_score(self, dataset, cycle_length, max_segments):
        """Compute the Bartels-style periodicity score for a candidate cycle.

        Parameters
        ----------
        dataset : sequence
            Detrended input values used to evaluate the candidate cycle.
        cycle_length : float
            Candidate period length expressed in samples.
        max_segments : int
            Maximum number of full cycle-length segments to include in the
            score calculation.

        Returns
        -------
        tuple
            ``(bartels_score, segment_count)``. A score close to zero indicates
            weaker periodic consistency; higher values indicate stronger
            periodic consistency under the legacy formula.
        """
        bartelsscore = 0
        segmentspassed = 0

        datacounter = 0
        A = 0
        B = 0
        SUM_A = 0
        SUM_B = 0
        SUM_A2B2 = 0
        bval = 0
        SI = 0
        CO = 0
        bogenmass = 0

        bval = 360 / cycle_length

        for x in range(len(dataset)):
            bogenmass = (bval * (x + 1)) / 180 * math.pi

            SI = math.sin(bogenmass) * dataset[x]
            CO = math.cos(bogenmass) * dataset[x]

            A += SI
            B += CO

            datacounter += 1

            if datacounter == int(cycle_length) and segmentspassed < int(max_segments):
                SUM_A += A
                SUM_B += B
                SUM_A2B2 += A ** 2 + B ** 2

                segmentspassed += 1
                datacounter = 0
                A = 0
                B = 0

        if segmentspassed != 0:
            SUM_A_Average = SUM_A / segmentspassed
            SUM_B_Average = SUM_B / segmentspassed
            Amplitude = math.sqrt(SUM_A_Average ** 2 + SUM_B_Average ** 2)

            SUM_A2B2_Average = SUM_A2B2 / segmentspassed
            Amplitude_A2B2 = math.sqrt(SUM_A2B2_Average)

            a1 = Amplitude_A2B2 / math.sqrt(segmentspassed)
            b1 = Amplitude / a1

            bartelsscore = 1 / math.exp(b1 ** 2)

        return bartelsscore, segmentspassed
