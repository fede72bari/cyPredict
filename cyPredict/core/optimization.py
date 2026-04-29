"""Optimization helpers for the legacy cyPredict workflows."""


class OptimizationMixin:
    """Small optimization utilities shared by genetic optimization paths."""

    def custom_crossover(self, ind1, ind2):
        """Apply the legacy DEAP crossover strategy.

        Parameters
        ----------
        ind1, ind2 : deap creator.Individual or sequence-like
            Individuals to cross over. When both contain more than one gene the
            historical two-point crossover is used. Single-gene individuals use
            uniform crossover with probability ``0.5``.

        Returns
        -------
        tuple
            The pair returned by the selected DEAP crossover operator.
        """
        from deap import tools

        if len(ind1) > 1 and len(ind2) > 1:
            return tools.cxTwoPoint(ind1, ind2)

        return tools.cxUniform(ind1, ind2, 0.5)
