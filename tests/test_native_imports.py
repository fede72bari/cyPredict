import importlib.util
import math

import numpy as np

from cyPredict.native_imports import (
    REQUIRED_CYGAOPT_ABI_VERSION,
    ensure_native_module_paths,
    native_module_dirs,
)


def test_native_modules_are_importable_from_repo_native_paths():
    ensure_native_module_paths()

    expected = {
        "goertzel": "native\\goertzel",
        "cyfitness": "native\\cyfitness",
        "cyGAopt": "native\\cygaopt",
        "cyGAoptMultiCore": "native\\cygaopt_multicore",
    }

    for module_name, expected_fragment in expected.items():
        spec = importlib.util.find_spec(module_name)
        assert spec is not None, module_name
        assert spec.origin is not None, module_name
        normalized_origin = spec.origin.replace("/", "\\")
        assert expected_fragment in normalized_origin


def test_native_modules_expose_expected_entrypoints():
    ensure_native_module_paths()

    import cyGAopt
    import cyGAoptMultiCore
    import cyfitness
    import goertzel

    assert hasattr(goertzel, "goertzel_general_shortened")
    assert hasattr(goertzel, "goertzel_DFT")
    assert hasattr(cyfitness, "evaluate_fitness")
    assert hasattr(cyGAopt, "run_genetic_algorithm")
    assert hasattr(cyGAoptMultiCore, "run_genetic_algorithm")
    assert hasattr(cyGAoptMultiCore, "evaluate_cycle_loss")
    assert cyGAopt.ABI_VERSION == REQUIRED_CYGAOPT_ABI_VERSION
    assert cyGAoptMultiCore.ABI_VERSION == REQUIRED_CYGAOPT_ABI_VERSION


def test_goertzel_native_smoke_call_returns_numeric_outputs():
    ensure_native_module_paths()

    import goertzel

    sample_count = 32
    cycle_length = 8.0
    data = np.sin(2.0 * np.pi * np.arange(sample_count, dtype=np.float64) / cycle_length)

    amp, phase, minoffset, minoffset2, maxoffset = goertzel.goertzel_DFT(data, cycle_length)
    assert amp > 0.0
    assert all(math.isfinite(float(value)) for value in (phase, minoffset, minoffset2, maxoffset))

    frequency_indexes = np.array([sample_count / cycle_length], dtype=np.float64)
    transform = goertzel.goertzel_general_shortened(data, frequency_indexes)
    assert transform.shape == (1,)
    assert np.iscomplexobj(transform)
    assert abs(transform[0]) > 0.0


def test_cyfitness_native_smoke_call_returns_finite_loss():
    ensure_native_module_paths()

    import cyfitness

    reference = np.array([0.0, 100.0, 0.0, -100.0] * 4, dtype=np.float64)
    individual = np.array([1.0, 0.25, 0.0], dtype=np.float64)
    cycles = [
        {
            "peak_frequencies": 0.25,
            "peak_phases": 0.0,
            "peak_periods": 4.0,
            "start_rebuilt_signal_index": 0,
        }
    ]

    loss = cyfitness.evaluate_fitness(
        individual,
        reference,
        cycles,
        1,
        1,
        len(reference),
        0,
        0,
        1.0,
        "mse",
        0,
    )

    assert math.isfinite(float(loss))


def test_cygaopt_multicore_native_loss_smoke_call_returns_finite_loss():
    ensure_native_module_paths()

    import cyGAoptMultiCore

    reference = [0.0, 100.0, 0.0, -100.0] * 4
    full_individual = [1.0, 0.25, 0.0]
    loss = cyGAoptMultiCore.evaluate_cycle_loss(
        full_individual,
        reference,
        [0.25],
        [0.0],
        [4.0],
        [0],
        frequencies_ft=True,
        phases_ft=True,
        fitness_type="mse",
    )

    assert math.isfinite(float(loss))


def test_native_build_directories_are_preferred_when_present():
    ensure_native_module_paths()

    existing_build_dirs = [path for path in native_module_dirs() if "\\build\\lib." in str(path)]
    if not existing_build_dirs:
        return

    first_existing_path = next(path for path in native_module_dirs() if path.exists())
    assert "\\build\\lib." in str(first_existing_path)
