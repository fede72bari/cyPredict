#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cygaopt_core {

inline double mean_squared_error(
    const std::vector<double>& left,
    const std::vector<double>& right,
    int start_index
) {
    const int length = static_cast<int>(left.size());
    if (start_index < 0 || start_index >= length) {
        throw std::runtime_error("invalid MSE start index");
    }

    double total = 0.0;
    int count = 0;
    for (int index = start_index; index < length; ++index) {
        const double diff = left[index] - right[index];
        total += diff * diff;
        ++count;
    }

    return count > 0 ? total / static_cast<double>(count) : 1e9;
}

inline double evaluate_cycle_loss(
    const std::vector<double>& full_individual,
    const std::vector<double>& reference_signal,
    const std::vector<double>& peak_frequencies,
    const std::vector<double>& peak_phases,
    const std::vector<double>& peak_periods,
    const std::vector<int>& start_indices,
    bool frequencies_ft,
    bool phases_ft,
    int best_fit_start_back_period,
    bool period_related_rebuild_range,
    double period_related_rebuild_multiplier,
    const std::string& fitness_type
) {
    const int n_cycles = static_cast<int>(peak_frequencies.size());
    const int len_series = static_cast<int>(reference_signal.size());

    if (n_cycles <= 0 || len_series <= 0) {
        return 1e9;
    }
    if (peak_phases.size() != static_cast<size_t>(n_cycles) ||
        peak_periods.size() != static_cast<size_t>(n_cycles) ||
        start_indices.size() != static_cast<size_t>(n_cycles)) {
        throw std::runtime_error("cycle metadata vectors must have matching sizes");
    }
    if (full_individual.size() != static_cast<size_t>(3 * n_cycles)) {
        throw std::runtime_error("full_individual size must be 3 * n_cycles");
    }

    std::vector<double> composite(reference_signal.size(), 0.0);

    for (int cycle = 0; cycle < n_cycles; ++cycle) {
        const double amplitude = full_individual[cycle];
        const double frequency = frequencies_ft ? full_individual[n_cycles + cycle] : peak_frequencies[cycle];
        const double phase = phases_ft ? full_individual[2 * n_cycles + cycle] : peak_phases[cycle];
        const double peak_period = peak_periods[cycle] > 0.0 ? peak_periods[cycle] :
            (frequency != 0.0 ? 1.0 / frequency : 1e6);

        int start = start_indices[cycle];
        if (start < 0) {
            start = 0;
        }
        if (start >= len_series) {
            continue;
        }

        int cutoff = 0;
        if (period_related_rebuild_range) {
            cutoff = static_cast<int>(len_series - peak_period * period_related_rebuild_multiplier);
            if (cutoff < 0) {
                cutoff = 0;
            }
        }

        for (int index = start; index < len_series; ++index) {
            if (period_related_rebuild_range && index < cutoff) {
                continue;
            }
            const double time_index = static_cast<double>(index - start);
            composite[index] += amplitude * std::sin(2.0 * M_PI * frequency * time_index + phase);
        }
    }

    int max_pos = 0;
    if (best_fit_start_back_period == 0) {
        for (int start : start_indices) {
            if (start > max_pos) {
                max_pos = start;
            }
        }
    } else {
        const int alternate_pos = len_series - best_fit_start_back_period;
        if (alternate_pos > max_pos) {
            max_pos = alternate_pos;
        }
    }

    if (max_pos < 0 || max_pos >= len_series) {
        max_pos = 0;
    }

    double composite_min = composite[max_pos];
    double composite_max = composite[max_pos];
    for (int index = max_pos + 1; index < len_series; ++index) {
        if (composite[index] < composite_min) {
            composite_min = composite[index];
        }
        if (composite[index] > composite_max) {
            composite_max = composite[index];
        }
    }

    const double composite_range = composite_max - composite_min;
    if (composite_range > 0.0) {
        for (int index = max_pos; index < len_series; ++index) {
            composite[index] = ((composite[index] - composite_min) / composite_range) * 2.0 - 1.0;
        }
    }
    for (int index = max_pos; index < len_series; ++index) {
        composite[index] *= 100.0;
    }

    if (fitness_type == "mse" || fitness_type.empty()) {
        return mean_squared_error(reference_signal, composite, max_pos);
    }

    return mean_squared_error(reference_signal, composite, max_pos);
}

}  // namespace cygaopt_core
