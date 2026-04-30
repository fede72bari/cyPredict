#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <vector>

#include "genetic_core.hpp"

namespace py = pybind11;

std::vector<double> run_genetic_algorithm(
    py::object fitness_func,
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
    bool initial_random_amplitudes = false,
    bool optimize_amplitudes = true,
    bool optimize_frequencies = true,
    bool optimize_phases = true,
    int seed = -1
) {
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
    options.n_cycles = n_cycles_opt.value_or(1);
    options.genetic_elitism = genetic_elitism;
    options.elitism_elements = elitism_elements;
    options.initial_random_amplitudes = initial_random_amplitudes;
    options.optimize_amplitudes = optimize_amplitudes;
    options.optimize_frequencies = optimize_frequencies;
    options.optimize_phases = optimize_phases;
    options.seed = seed;
    options.threads = 1;

    auto objective = [&](const std::vector<double>& individual) {
        py::gil_scoped_acquire acquire;
        py::object score = fitness_func(individual);
        return score.cast<double>();
    };

    return cygaopt_core::run_genetic_algorithm_minimize(objective, options);
}

PYBIND11_MODULE(cyGAopt, m) {
    m.attr("ABI_VERSION") = cygaopt_core::ABI_VERSION;
    m.attr("__version__") = "2.0.0";
    m.def(
        "run_genetic_algorithm",
        &run_genetic_algorithm,
        py::arg("fitness_func"),
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
        py::arg("seed") = -1
    );
}
