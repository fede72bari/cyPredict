#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include <algorithm>
#include <optional>
#include <cmath>
#include <iostream>

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
    int steps_n = 100
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_01(0.0, 1.0);
    std::uniform_int_distribution<> step_dist(0, steps_n - 1);
    std::normal_distribution<> dis_gauss(0.5 * steps_n, 2.5 * steps_n);

    auto generate_discrete_value = [&](int i) -> double {
        if (lb && ub) {
            double l = (*lb)[i];
            double u = (*ub)[i];
            double step_size = (u - l) / (steps_n - 1);
            int k = step_dist(gen);
            return l + k * step_size;
        } else {
            return dis_01(gen);
        }
    };

    auto gaussian_generate_value = [&](int i) -> double {
        if (lb && ub) {
            double l = (*lb)[i];
            double u = (*ub)[i];
            double step_size = (u - l) / (steps_n - 1);
            int k = std::clamp(static_cast<int>(std::round(dis_gauss(gen))), 0, steps_n - 1);
            return l + k * step_size;
        } else {
            return dis_01(gen);
        }
    };

    std::vector<std::vector<double>> population(population_n, std::vector<double>(gene_length));
    for (auto& individual : population) {
        for (int i = 0; i < gene_length; ++i) {
            individual[i] = gaussian_generate_value(i);
        }
    }

    std::vector<double> best_individual;
    double best_fitness = -1e9;
    double prev_best_fitness = -1e9;
    int stagnation_counter = 0;

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

        std::cout << "Generation " << gen_num << " best fitness: " << best_fitness << std::endl;

        if (best_fitness < prev_best_fitness + 1e-6) {
            stagnation_counter++;
        } else {
            stagnation_counter = 0;
        }
        prev_best_fitness = best_fitness;

        std::vector<std::vector<double>> new_population;

        for (int i = 0; i < 2; ++i)
            new_population.push_back(scored[i].second);

        if (stagnation_counter >= 20) {
            for (int i = 0; i < population_n / 2; ++i) {
                std::vector<double> new_individual(gene_length);
                for (int j = 0; j < gene_length; ++j) {
                    new_individual[j] = gaussian_generate_value(j);
                }
                new_population.push_back(new_individual);
            }
        }

        while (new_population.size() < population_n) {
            if ((new_population.size() % 10) == 0) {
                std::vector<double> rnd(gene_length);
                for (int j = 0; j < gene_length; ++j)
                    rnd[j] = gaussian_generate_value(j);
                new_population.push_back(rnd);
                continue;
            }

            const auto& p1 = scored[gen() % population_n].second;
            const auto& p2 = scored[gen() % population_n].second;
            std::vector<double> c1 = p1;
            std::vector<double> c2 = p2;

            for (int j = 0; j < gene_length; ++j) {
                if (dis_01(gen) < CXPB) {
                    std::swap(c1[j], c2[j]);
                }

                double step_size = (lb && ub) ? ((*ub)[j] - (*lb)[j]) / (steps_n - 1) : 0.01;
                int jump = (gen() % 4 == 0) ? 10 : 1;
                if (dis_01(gen) < MUTPB) {
                    int sign = (gen() % 2 == 0) ? 1 : -1;
                    c1[j] += sign * jump * step_size;
                }
                if (dis_01(gen) < MUTPB) {
                    int sign = (gen() % 2 == 0) ? 1 : -1;
                    c2[j] += sign * jump * step_size;
                }

                if (lb && ub) {
                    c1[j] = std::clamp(c1[j], (*lb)[j], (*ub)[j]);
                    c2[j] = std::clamp(c2[j], (*lb)[j], (*ub)[j]);
                }
            }

            new_population.push_back(c1);
            if (new_population.size() < population_n)
                new_population.push_back(c2);
        }
        

        population = std::move(new_population);
    }

    return best_individual;
}

PYBIND11_MODULE(cyGAopt, m) {
    m.def("run_genetic_algorithm", &run_genetic_algorithm, "Run Genetic Algorithm");
}
