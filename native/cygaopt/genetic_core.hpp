#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cygaopt_core {

constexpr int ABI_VERSION = 2;

struct GeneticOptions {
    int population_n = 0;
    double crossover_probability = 0.0;
    double mutation_probability = 0.0;
    int generations = 0;
    int gene_length = 0;
    std::optional<std::vector<double>> lower_bounds;
    std::optional<std::vector<double>> upper_bounds;
    int steps_n = 100;
    std::optional<std::vector<double>> initial_vector;
    int n_cycles = 1;
    bool genetic_elitism = true;
    int elitism_elements = 10;
    bool initial_random_amplitudes = false;
    bool optimize_amplitudes = true;
    bool optimize_frequencies = true;
    bool optimize_phases = true;
    int seed = -1;
    int threads = 1;
};

inline void validate_options(const GeneticOptions& options) {
    if (options.population_n <= 0) {
        throw std::runtime_error("population_n must be positive");
    }
    if (options.gene_length <= 0) {
        throw std::runtime_error("gene_length must be positive");
    }
    if (options.n_cycles <= 0) {
        throw std::runtime_error("n_cycles must be positive");
    }
    if (options.steps_n < 2) {
        throw std::runtime_error("steps_n must be at least 2");
    }
    if (options.lower_bounds && options.lower_bounds->size() != static_cast<size_t>(options.gene_length)) {
        throw std::runtime_error("lower_bounds size must match gene_length");
    }
    if (options.upper_bounds && options.upper_bounds->size() != static_cast<size_t>(options.gene_length)) {
        throw std::runtime_error("upper_bounds size must match gene_length");
    }
    if (static_cast<bool>(options.lower_bounds) != static_cast<bool>(options.upper_bounds)) {
        throw std::runtime_error("lower_bounds and upper_bounds must be provided together");
    }
    if (options.initial_vector && options.initial_vector->size() != static_cast<size_t>(options.gene_length)) {
        throw std::runtime_error("initial_vector size must match gene_length");
    }
}

inline double discrete_value(
    int index,
    const GeneticOptions& options,
    std::mt19937& generator,
    std::uniform_int_distribution<int>& step_distribution,
    std::uniform_real_distribution<double>& unit_distribution
) {
    if (options.lower_bounds && options.upper_bounds) {
        const double lower = (*options.lower_bounds)[index];
        const double upper = (*options.upper_bounds)[index];
        if (upper <= lower) {
            return lower;
        }
        const double step_size = (upper - lower) / static_cast<double>(options.steps_n - 1);
        return lower + step_distribution(generator) * step_size;
    }
    return unit_distribution(generator);
}

inline void maybe_mutate_block(
    std::vector<double>& individual,
    int offset,
    bool enabled,
    const GeneticOptions& options,
    std::mt19937& generator,
    std::uniform_real_distribution<double>& unit_distribution,
    std::uniform_int_distribution<int>& step_distribution
) {
    if (!enabled || offset >= options.gene_length) {
        return;
    }
    if (unit_distribution(generator) >= options.mutation_probability) {
        return;
    }

    const int end = std::min(options.gene_length, offset + options.n_cycles);
    for (int index = offset; index < end; ++index) {
        individual[index] = discrete_value(index, options, generator, step_distribution, unit_distribution);
    }
}

inline std::vector<double> tournament_select(
    const std::vector<std::pair<double, std::vector<double>>>& evaluated,
    std::mt19937& generator,
    int tournament_size = 3
) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(evaluated.size()) - 1);
    auto best = evaluated[pick(generator)];
    for (int index = 1; index < tournament_size; ++index) {
        auto candidate = evaluated[pick(generator)];
        if (candidate.first < best.first) {
            best = std::move(candidate);
        }
    }
    return best.second;
}

inline std::vector<double> run_genetic_algorithm_minimize(
    const std::function<double(const std::vector<double>&)>& objective,
    GeneticOptions options
) {
    validate_options(options);

    std::random_device random_device;
    std::mt19937 generator(options.seed >= 0 ? static_cast<unsigned int>(options.seed) : random_device());
    std::uniform_real_distribution<double> unit_distribution(0.0, 1.0);
    std::uniform_int_distribution<int> step_distribution(0, options.steps_n - 1);

    std::vector<std::vector<double>> population;
    population.reserve(options.population_n);

    for (int row = 0; row < options.population_n; ++row) {
        std::vector<double> individual(options.gene_length);

        for (int gene = 0; gene < options.gene_length; ++gene) {
            if (options.initial_vector) {
                individual[gene] = (*options.initial_vector)[gene];
            } else {
                individual[gene] = discrete_value(gene, options, generator, step_distribution, unit_distribution);
            }
        }

        if (options.initial_random_amplitudes || !options.initial_vector) {
            const int end = std::min(options.gene_length, options.n_cycles);
            for (int gene = 0; gene < end; ++gene) {
                individual[gene] = discrete_value(gene, options, generator, step_distribution, unit_distribution);
            }
        }

        population.push_back(std::move(individual));
    }

    std::vector<double> hall_of_fame;
    double hall_of_fame_loss = std::numeric_limits<double>::infinity();
    const int block_size = std::max(1, options.n_cycles);

    auto evaluate_population = [&](const std::vector<std::vector<double>>& current_population) {
        std::vector<std::pair<double, std::vector<double>>> evaluated(current_population.size());

#ifdef _OPENMP
        if (options.threads > 1) {
            const int previous_threads = omp_get_max_threads();
            omp_set_num_threads(options.threads);
#pragma omp parallel for
            for (int index = 0; index < static_cast<int>(current_population.size()); ++index) {
                evaluated[index] = std::make_pair(objective(current_population[index]), current_population[index]);
            }
            omp_set_num_threads(previous_threads);
        } else
#endif
        {
            for (size_t index = 0; index < current_population.size(); ++index) {
                evaluated[index] = std::make_pair(objective(current_population[index]), current_population[index]);
            }
        }

        std::sort(evaluated.begin(), evaluated.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        return evaluated;
    };

    for (int generation = 0; generation <= options.generations; ++generation) {
        auto evaluated = evaluate_population(population);
        if (!evaluated.empty() && evaluated.front().first < hall_of_fame_loss) {
            hall_of_fame_loss = evaluated.front().first;
            hall_of_fame = evaluated.front().second;
        }

        if (generation == options.generations) {
            break;
        }

        std::vector<std::vector<double>> offspring;
        offspring.reserve(options.population_n);

        if (options.genetic_elitism) {
            const int elite_count = std::min(options.elitism_elements, static_cast<int>(evaluated.size()));
            for (int index = 0; index < elite_count; ++index) {
                offspring.push_back(evaluated[index].second);
            }
        }

        while (static_cast<int>(offspring.size()) < options.population_n) {
            auto parent1 = tournament_select(evaluated, generator);
            auto parent2 = tournament_select(evaluated, generator);
            auto child1 = parent1;
            auto child2 = parent2;

            for (int offset = 0; offset < options.gene_length; offset += block_size) {
                if (unit_distribution(generator) < options.crossover_probability) {
                    const int end = std::min(options.gene_length, offset + block_size);
                    for (int gene = offset; gene < end; ++gene) {
                        std::swap(child1[gene], child2[gene]);
                    }
                }
            }

            maybe_mutate_block(
                child1, 0, options.optimize_amplitudes, options, generator, unit_distribution, step_distribution);
            maybe_mutate_block(
                child1, options.n_cycles, options.optimize_frequencies, options, generator, unit_distribution, step_distribution);
            maybe_mutate_block(
                child1, 2 * options.n_cycles, options.optimize_phases, options, generator, unit_distribution, step_distribution);

            maybe_mutate_block(
                child2, 0, options.optimize_amplitudes, options, generator, unit_distribution, step_distribution);
            maybe_mutate_block(
                child2, options.n_cycles, options.optimize_frequencies, options, generator, unit_distribution, step_distribution);
            maybe_mutate_block(
                child2, 2 * options.n_cycles, options.optimize_phases, options, generator, unit_distribution, step_distribution);

            offspring.push_back(std::move(child1));
            if (static_cast<int>(offspring.size()) < options.population_n) {
                offspring.push_back(std::move(child2));
            }
        }

        population = std::move(offspring);
    }

    return hall_of_fame;
}

}  // namespace cygaopt_core
