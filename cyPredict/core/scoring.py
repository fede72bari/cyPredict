"""Ranking helpers for cyPredict candidate cycles."""

import pandas as pd


class ScoringMixin:
    """Compute legacy row and global scores."""

    def get_row_score(self, row):
        """Return the sum of per-column rank scores for one row."""
        scores = row.sum()

        return scores

    def get_gloabl_score(self, data, ascending_columns, descending_columns):
        """Compute the legacy global score over ascending and descending ranks.

        The misspelled method name is preserved for compatibility with existing
        notebooks and internal calls.
        """
        df = pd.DataFrame(data)

        data_ascending = df[ascending_columns].rank(ascending=True, axis=0)
        data_descending = df[descending_columns].rank(ascending=False, axis=0)

        global_score = pd.concat([data_ascending, data_descending], axis=1)
        global_score['global_score'] = global_score.apply(self.get_row_score, axis=1)

        return global_score['global_score']
