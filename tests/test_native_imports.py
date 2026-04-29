import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_PATHS = [
    REPO_ROOT / "native" / "goertzel",
    REPO_ROOT / "native" / "cyfitness",
    REPO_ROOT / "native" / "cygaopt",
    REPO_ROOT / "native" / "cygaopt_multicore",
]


def test_native_modules_are_importable_from_repo_native_paths():
    for path in reversed(NATIVE_PATHS):
        sys.path.insert(0, str(path))

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
    for path in reversed(NATIVE_PATHS):
        sys.path.insert(0, str(path))

    import cyGAopt
    import cyGAoptMultiCore
    import cyfitness
    import goertzel

    assert hasattr(goertzel, "goertzel_general_shortened")
    assert hasattr(goertzel, "goertzel_DFT")
    assert hasattr(cyfitness, "evaluate_fitness")
    assert hasattr(cyGAopt, "run_genetic_algorithm")
    assert hasattr(cyGAoptMultiCore, "run_genetic_algorithm")

