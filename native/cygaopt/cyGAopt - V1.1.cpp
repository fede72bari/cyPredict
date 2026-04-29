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

    // Funzione di mutazione e inizializzazione
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

    // Inizializzazione della popolazione
    std::vector<std::vector<double>> population(population_n, std::vector<double>(gene_length));
    for (auto& individual : population)
        for (int i = 0; i < gene_length; ++i)
            individual[i] = generate_discrete_value(i);

    std::vector<double> best_individual;
    double best_fitness = -1e9;
/*
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
                if (dis_01(gen) < CXPB)
                    child[j] = p2[j];

                if (dis_01(gen) < MUTPB)
                    child[j] = generate_discrete_value(j);  // MUTAZIONE DISCRETIZZATA
            }

            new_population.push_back(child);
            new_population.push_back(p2); // elitismo
        }

        population = std::move(new_population);
    }
    */
    /*
    int N_elite = 2;
    int N_random = 2;
    
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
    
        // Elite preservation
        std::vector<std::vector<double>> elite;
        for (int i = 0; i < std::min(N_elite, population_n); ++i)
            elite.push_back(scored[i].second);
    
        std::vector<std::vector<double>> new_population;
    
        // Crossover a blocchi + mutazione per ogni coppia
        for (int i = 0; i < (population_n - N_elite - N_random) / 2; ++i) {
            const auto& p1 = scored[i % population_n].second;
            const auto& p2 = scored[(i + 1) % population_n].second;
    
            std::vector<double> child1 = p1;
            std::vector<double> child2 = p2;
    
            for (int h = 0; h < gene_length / 3; ++h) {
                int base = h * 3;
                if (dis_01(gen) < CXPB) {
                    // blocco crossover
                    child1[base]     = p2[base];
                    child1[base + 1] = p2[base + 1];
                    child1[base + 2] = p2[base + 2];
    
                    child2[base]     = p1[base];
                    child2[base + 1] = p1[base + 1];
                    child2[base + 2] = p1[base + 2];
                }
    
                if (dis_01(gen) < MUTPB)
                    child1[base] = generate_discrete_value(base);
                if (dis_01(gen) < MUTPB)
                    child1[base + 1] = generate_discrete_value(base + 1);
                if (dis_01(gen) < MUTPB)
                    child1[base + 2] = generate_discrete_value(base + 2);
    
                if (dis_01(gen) < MUTPB)
                    child2[base] = generate_discrete_value(base);
                if (dis_01(gen) < MUTPB)
                    child2[base + 1] = generate_discrete_value(base + 1);
                if (dis_01(gen) < MUTPB)
                    child2[base + 2] = generate_discrete_value(base + 2);
            }
    
            new_population.push_back(child1);
            new_population.push_back(child2);
        }
    
        // Inserisci gli elite veri
        for (const auto& elite_ind : elite)
            new_population.push_back(elite_ind);
    
        // Inserisci individui random
        for (int i = 0; i < N_random; ++i) {
            std::vector<double> rnd(gene_length);
            for (int j = 0; j < gene_length; ++j)
                rnd[j] = generate_discrete_value(j);
            new_population.push_back(rnd);
        }
    
        // Trim (in caso di eccedenza)
        while (new_population.size() > static_cast<size_t>(population_n))
            new_population.pop_back();
    
        population = std::move(new_population);
    }
    */


/*
    for (int gen_num = 0; gen_num < NGEN; ++gen_num) {
        std::vector<std::pair<double, std::vector<double>>> scored;
        for (auto& individual : population) {
            double score = fitness_func(individual).cast<double>();
            scored.emplace_back(score, individual);
        }
    
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            return a.first < b.first; // minimizzazione
        });
    
        // 📊 Stampa leggibile dei top 3
        // std::cout << "\n====== GENERAZIONE " << gen_num + 1 << " ======\n";
        // for (int i = 0; i < std::min(3, population_n); ++i) {
        //     std::cout << "🏅 indiv " << i << ": ";
        //     for (size_t j = 0; j < scored[i].second.size(); j += 3)
        //         std::cout << "[a:" << scored[i].second[j]
        //                   << " f:" << scored[i].second[j + 1]
        //                   << " p:" << scored[i].second[j + 2] << "] ";
        //     std::cout << "| fitness = " << scored[i].first << "\n";
        // }
        // std::cout << std::flush;
    
        best_individual = scored[0].second;
        best_fitness = scored[0].first;
    
        std::vector<std::vector<double>> new_population;
    
        // 🎲 Elitismo
        new_population.push_back(scored[0].second);
        new_population.push_back(scored[1].second);
    
        while (new_population.size() < population_n) {
            const auto& p1 = scored[gen() % (population_n / 2)].second;
            const auto& p2 = scored[gen() % (population_n / 2)].second;
    
            std::vector<double> child1(gene_length);
            std::vector<double> child2(gene_length);
    
            for (int j = 0; j < gene_length; ++j) {
                child1[j] = (dis_01(gen) < 0.5) ? p1[j] : p2[j];
                child2[j] = (dis_01(gen) < 0.5) ? p2[j] : p1[j];
    
                // 💥 Mutazioni (salto ±10 step)
                if (dis_01(gen) < MUTPB) {
                    if (dis_01(gen) < 0.2) {
                        int offset = (gen() % 21 - 10);
                        int current = std::round((child1[j] - (*lb)[j]) / ((*ub)[j] - (*lb)[j]) * steps_n);
                        current = std::max(0, std::min(steps_n, current + offset));
                        child1[j] = (*lb)[j] + current * ((*ub)[j] - (*lb)[j]) / steps_n;
                    } else {
                        child1[j] = generate_discrete_value(j);
                    }
                }
    
                if (dis_01(gen) < MUTPB) {
                    if (dis_01(gen) < 0.2) {
                        int offset = (gen() % 21 - 10);
                        int current = std::round((child2[j] - (*lb)[j]) / ((*ub)[j] - (*lb)[j]) * steps_n);
                        current = std::max(0, std::min(steps_n, current + offset));
                        child2[j] = (*lb)[j] + current * ((*ub)[j] - (*lb)[j]) / steps_n;
                    } else {
                        child2[j] = generate_discrete_value(j);
                    }
                }
            }
    
            new_population.push_back(child1);
            if (new_population.size() < population_n)
                new_population.push_back(child2);
    
            // 🧪 Individuo totalmente random ogni 10
            if (new_population.size() % 10 == 0 && new_population.size() < population_n) {
                std::vector<double> random_indiv(gene_length);
                for (int j = 0; j < gene_length; ++j)
                    random_indiv[j] = generate_discrete_value(j);
                new_population.push_back(random_indiv);
            }
        }
    
        population = std::move(new_population);
    }


    return best_individual;
}

PYBIND11_MODULE(cyGAopt, m) {
    m.def("run_genetic_algorithm", &run_genetic_algorithm,
        py::arg("fitness_func"),
        py::arg("population_n"),
        py::arg("CXPB"),
        py::arg("MUTPB"),
        py::arg("NGEN"),
        py::arg("gene_length"),
        py::arg("lb") = std::nullopt,
        py::arg("ub") = std::nullopt,
        py::arg("steps_n") = 100,
        "Esegue un algoritmo genetico classico con vincoli opzionali e range discretizzati");
}
