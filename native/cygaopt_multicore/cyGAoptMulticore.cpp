#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "../cygaopt/cycle_fitness_core.hpp"
#include "../cygaopt/genetic_core.hpp"

namespace py = pybind11;

std::vector<double> default_peak_periods(const std::vector<double>& frequencies) {
    std::vector<double> periods;
    periods.reserve(frequencies.size());
    for (double frequency : frequencies) {
        periods.push_back(frequency != 0.0 ? 1.0 / frequency : 1e6);
    }
    return periods;
}

std::vector<int> default_start_indices(int n_cycles, int start_rebuild_index) {
    return std::vector<int>(static_cast<size_t>(n_cycles), start_rebuild_index);
}

double evaluate_cycle_loss(
    const std::vector<double>& full_individual,
    const std::vector<double>& reference_signal,
    const std::vector<double>& peak_frequencies,
    const std::vector<double>& peak_phases,
    const std::vector<double>& peak_periods,
    const std::vector<int>& start_indices,
    bool frequencies_ft = true,
    bool phases_ft = true,
    int best_fit_start_back_period = 0,
    bool period_related_rebuild_range = false,
    double period_related_rebuild_multiplier = 1.0,
    const std::string& fitness_type = "mse"
) {
    return cygaopt_core::evaluate_cycle_loss(
        full_individual,
        reference_signal,
        peak_frequencies,
        peak_phases,
        peak_periods,
        start_indices,
        frequencies_ft,
        phases_ft,
        best_fit_start_back_period,
        period_related_rebuild_range,
        period_related_rebuild_multiplier,
        fitness_type
    );
}

std::vector<double> run_genetic_algorithm(
    const std::vector<double>& reference_signal,
    int population_n,
    double CXPB,
    double MUTPB,
    int NGEN,
    int gene_length,
    std::optional<std::vector<double>> lb = std::nullopt,
    std::optional<std::vector<double>> ub = std::nullopt,
    int steps_n = 100,
    std::optional<std::vector<double>> initial_vector_opt = std::nullopt,
    std::optional<int> n_cycles_opt = std::nullopt,
    bool genetic_elitism = true,
    int elitism_elements = 10,
    std::optional<bool> initial_random_amplitudes_opt = std::nullopt,
    bool optimize_amplitudes = true,
    bool optimize_frequencies = true,
    bool optimize_phases = true,
    int start_rebuild_index = 0,
    bool period_related_rebuild_range = false,
    double period_multiplier = 1.0,
    std::optional<std::vector<double>> peak_frequencies_opt = std::nullopt,
    std::optional<std::vector<double>> peak_phases_opt = std::nullopt,
    std::optional<std::vector<double>> peak_periods_opt = std::nullopt,
    std::optional<std::vector<int>> start_indices_opt = std::nullopt,
    int best_fit_start_back_period = 0,
    const std::string& fitness_type = "mse",
    int seed = -1,
    int threads = 0
) {
    const int n_cycles = n_cycles_opt.value_or(1);

    if (!initial_vector_opt || initial_vector_opt->size() != static_cast<size_t>(gene_length)) {
        throw std::runtime_error("initial_vector_opt must be provided with correct size");
    }
    if (gene_length != 3 * n_cycles) {
        throw std::runtime_error("cyGAoptMultiCore expects full vectors with 3 * n_cycles genes");
    }

    std::vector<double> peak_frequencies = peak_frequencies_opt.value_or(
        std::vector<double>(initial_vector_opt->begin() + n_cycles, initial_vector_opt->begin() + 2 * n_cycles));
    std::vector<double> peak_phases = peak_phases_opt.value_or(
        std::vector<double>(initial_vector_opt->begin() + 2 * n_cycles, initial_vector_opt->begin() + 3 * n_cycles));
    std::vector<double> peak_periods = peak_periods_opt.value_or(default_peak_periods(peak_frequencies));
    std::vector<int> start_indices = start_indices_opt.value_or(default_start_indices(n_cycles, start_rebuild_index));

    cygaopt_core::GeneticOptions options;
    options.population_n = population_n;
    options.crossover_probability = CXPB;
    options.mutation_probability = MUTPB;
    options.generations = NGEN;
    options.gene_length = gene_length;
    options.lower_bounds = lb;
    options.upper_bounds = ub;
    options.steps_n = steps_n;
    options.initial_vector = initial_vector_opt;
    options.n_cycles = n_cycles;
    options.genetic_elitism = genetic_elitism;
    options.elitism_elements = elitism_elements;
    options.initial_random_amplitudes = initial_random_amplitudes_opt.value_or(false);
    options.optimize_amplitudes = optimize_amplitudes;
    options.optimize_frequencies = optimize_frequencies;
    options.optimize_phases = optimize_phases;
    options.seed = seed;
    options.threads = threads > 0 ? threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    auto objective = [&](const std::vector<double>& individual) {
        return cygaopt_core::evaluate_cycle_loss(
            individual,
            reference_signal,
            peak_frequencies,
            peak_phases,
            peak_periods,
            start_indices,
            optimize_frequencies,
            optimize_phases,
            best_fit_start_back_period,
            period_related_rebuild_range,
            period_multiplier,
            fitness_type
        );
    };

    return cygaopt_core::run_genetic_algorithm_minimize(objective, options);
}

PYBIND11_MODULE(cyGAoptMultiCore, m) {
    m.attr("ABI_VERSION") = cygaopt_core::ABI_VERSION;
    m.attr("__version__") = "2.0.0";
    m.def(
        "evaluate_cycle_loss",
        &evaluate_cycle_loss,
        py::arg("full_individual"),
        py::arg("reference_signal"),
        py::arg("peak_frequencies"),
        py::arg("peak_phases"),
        py::arg("peak_periods"),
        py::arg("start_indices"),
        py::arg("frequencies_ft") = true,
        py::arg("phases_ft") = true,
        py::arg("best_fit_start_back_period") = 0,
        py::arg("period_related_rebuild_range") = false,
        py::arg("period_related_rebuild_multiplier") = 1.0,
        py::arg("fitness_type") = "mse"
    );
    m.def(
        "run_genetic_algorithm",
        &run_genetic_algorithm,
        py::arg("reference_signal"),
        py::arg("population_n"),
        py::arg("CXPB"),
        py::arg("MUTPB"),
        py::arg("NGEN"),
        py::arg("gene_length"),
        py::arg("lb") = std::nullopt,
        py::arg("ub") = std::nullopt,
        py::arg("steps_n") = 100,
        py::arg("initial_vector_opt") = std::nullopt,
        py::arg("n_cycles_opt") = std::nullopt,
        py::arg("genetic_elitism") = true,
        py::arg("elitism_elements") = 10,
        py::arg("initial_random_amplitudes") = false,
        py::arg("optimize_amplitudes") = true,
        py::arg("optimize_frequencies") = true,
        py::arg("optimize_phases") = true,
        py::arg("start_rebuild_index") = 0,
        py::arg("period_related_rebuild_range") = false,
        py::arg("period_multiplier") = 1.0,
        py::arg("peak_frequencies") = std::nullopt,
        py::arg("peak_phases") = std::nullopt,
        py::arg("peak_periods") = std::nullopt,
        py::arg("start_indices") = std::nullopt,
        py::arg("best_fit_start_back_period") = 0,
        py::arg("fitness_type") = "mse",
        py::arg("seed") = -1,
        py::arg("threads") = 0
    );
}
