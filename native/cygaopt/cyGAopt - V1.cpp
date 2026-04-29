#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include <algorithm>

namespace py = pybind11;

//std::vector<double> run_genetic_algorithm(py::object fitness_func,
//    int population_n,
//    double CXPB,
//    double MUTPB,
//    int NGEN,
//    int gene_length) {
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> dis(0.0, 1.0);

#include <optional>  // Per std::optional

std::vector<double> run_genetic_algorithm(
    py::object fitness_func,
    int population_n,
    double CXPB,
    double MUTPB,
    int NGEN,
    int gene_length,
    std::optional<std::vector<double>> lb = std::nullopt,
    std::optional<std::vector<double>> ub = std::nullopt
);

    // Inizializzazione della popolazione
    std::vector<std::vector<double>> population(population_n, std::vector<double>(gene_length));
    for (auto& individual : population)
        for (auto& gene : individual)
            gene = dis(gen);

    std::vector<double> best_individual;
    double best_fitness = -1e9;

    for (int gen_num = 0; gen_num < NGEN; ++gen_num) {
        std::vector<std::pair<double, std::vector<double>>> scored;
        for (auto& individual : population) {
            double score = fitness_func(individual).cast<double>();
            scored.emplace_back(score, individual);
        }

        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
            });

        best_individual = scored[0].second;
        best_fitness = scored[0].first;

        std::vector<std::vector<double>> new_population;

        for (int i = 0; i < population_n / 2; ++i) {
            const auto& p1 = scored[i % population_n].second;
            const auto& p2 = scored[(i + 1) % population_n].second;

            std::vector<double> child = p1;

            for (int j = 0; j < gene_length; ++j) {
                if (dis(gen) < CXPB)
                    child[j] = p2[j];
                if (dis(gen) < MUTPB)
                    child[j] = dis(gen);
            }

            new_population.push_back(child);
            new_population.push_back(p2); // elitismo
        }

        population = std::move(new_population);
    }

    return best_individual;
}

PYBIND11_MODULE(cyGAopt, m) {
    m.def("run_genetic_algorithm", &run_genetic_algorithm, "Esegue un algoritmo genetico classico",
        py::arg("fitness_func"),
        py::arg("population_n"),
        py::arg("CXPB"),
        py::arg("MUTPB"),
        py::arg("NGEN"),
        py::arg("gene_length"));
}
