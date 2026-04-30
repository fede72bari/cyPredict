import math

import numpy as np

from cyPredict.native_imports import ensure_native_module_paths


ensure_native_module_paths()

import cyGAopt
import cyGAoptMultiCore
from cyfitness import evaluate_fitness


def cyfitness_loss(full_individual, reference, cycles, frequencies_ft, phases_ft):
    n_cycles = len(cycles)
    active = list(full_individual[:n_cycles])
    if frequencies_ft:
        active += list(full_individual[n_cycles:2 * n_cycles])
    if phases_ft:
        active += list(full_individual[2 * n_cycles:3 * n_cycles])

    return evaluate_fitness(
        np.asarray(active, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        cycles,
        int(frequencies_ft),
        int(phases_ft),
        len(reference),
        0,
        1,
        1.5,
        "mse",
        0,
    )


def native_multicore_loss(full_individual, reference, cycles, frequencies_ft, phases_ft):
    return cyGAoptMultiCore.evaluate_cycle_loss(
        list(full_individual),
        list(reference),
        [cycle["peak_frequencies"] for cycle in cycles],
        [cycle["peak_phases"] for cycle in cycles],
        [cycle["peak_periods"] for cycle in cycles],
        [cycle["start_rebuilt_signal_index"] for cycle in cycles],
        frequencies_ft,
        phases_ft,
        0,
        True,
        1.5,
        "mse",
    )


def test_native_multicore_fitness_matches_cyfitness_for_flag_combinations():
    reference = np.linspace(-80.0, 95.0, 48)
    full_individual = [0.8, 1.1, 0.055, 0.082, 0.15, -0.35]
    cycles = [
        {
            "peak_frequencies": 0.05,
            "peak_phases": 0.1,
            "peak_periods": 20.0,
            "start_rebuilt_signal_index": 4,
        },
        {
            "peak_frequencies": 0.08,
            "peak_phases": -0.4,
            "peak_periods": 12.5,
            "start_rebuilt_signal_index": 7,
        },
    ]

    for frequencies_ft in (False, True):
        for phases_ft in (False, True):
            expected = cyfitness_loss(full_individual, reference, cycles, frequencies_ft, phases_ft)
            actual = native_multicore_loss(full_individual, reference, cycles, frequencies_ft, phases_ft)
            assert actual == pytest_approx(expected)


def test_single_core_ga_minimizes_positive_loss_with_seed():
    def objective(individual):
        return sum((value - 0.25) ** 2 for value in individual)

    kwargs = dict(
        fitness_func=objective,
        population_n=32,
        CXPB=0.7,
        MUTPB=0.4,
        NGEN=40,
        gene_length=3,
        lb=[0.0, 0.0, 0.0],
        ub=[1.0, 1.0, 1.0],
        steps_n=200,
        initial_vector_opt=[0.5, 0.5, 0.5],
        n_cycles_opt=1,
        initial_random_amplitudes=True,
        optimize_amplitudes=True,
        optimize_frequencies=True,
        optimize_phases=True,
        seed=123,
    )

    first = cyGAopt.run_genetic_algorithm(**kwargs)
    second = cyGAopt.run_genetic_algorithm(**kwargs)

    assert first == second
    assert objective(first) < objective([0.5, 0.5, 0.5])
    assert math.isfinite(objective(first))


def pytest_approx(value):
    import pytest

    return pytest.approx(value, rel=1e-12, abs=1e-12)
