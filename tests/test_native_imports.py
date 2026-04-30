import importlib.util

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


def test_native_build_directories_are_preferred_when_present():
    ensure_native_module_paths()

    existing_build_dirs = [path for path in native_module_dirs() if "\\build\\lib." in str(path)]
    if not existing_build_dirs:
        return

    first_existing_path = next(path for path in native_module_dirs() if path.exists())
    assert "\\build\\lib." in str(first_existing_path)
