// goertzel.cpp — compatibile con NumPy 2.x e Pybind11 >= 2.12
// V1.3

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

py::array_t<std::complex<double>> goertzel_general_shortened(py::array_t<double> x, py::array_t<double> indvec) {
    auto x_buf = x.unchecked<1>();
    auto indvec_buf = indvec.unchecked<1>();

    size_t lx = x_buf.shape(0);
    size_t no_freq = indvec_buf.shape(0);

    auto result = py::array_t<std::complex<double>>(no_freq);
    auto result_buf = result.mutable_unchecked<1>();

    for (size_t cnt_freq = 0; cnt_freq < no_freq; cnt_freq++) {
        double pik_term = 2.0 * M_PI * indvec_buf(cnt_freq) / lx;
        double cos_pik_term2 = 2.0 * std::cos(pik_term);
        double cc_real = std::cos(-pik_term);
        double cc_imag = std::sin(-pik_term);

        double s0 = 0.0, s1 = 0.0, s2 = 0.0;

        for (size_t ind = 0; ind < lx; ind++) {
            s0 = x_buf(ind) + cos_pik_term2 * s1 - s2;
            s2 = s1;
            s1 = s0;
        }

        s0 = x_buf(lx - 1) + cos_pik_term2 * s1 - s2;

        double y_real = s0 - s1 * cc_real;
        double y_imag = -s1 * cc_imag;

        result_buf(cnt_freq) = std::complex<double>(y_real, y_imag);
    }

    return result;
}

py::tuple goertzel_DFT(py::array_t<double> testdata, double testcycle_length, bool debug = false) {
    auto data_buf = testdata.unchecked<1>();
    size_t N = data_buf.shape(0);

    double f = 1.0 / testcycle_length;
    double coeff = 2.0 * std::cos(2.0 * M_PI * f);
    double Q0 = 0, Q1 = 0, Q2 = 0;

    for (size_t i = 0; i < N; i++) {
        Q0 = coeff * Q1 - Q2 + data_buf(i);
        Q2 = Q1;
        Q1 = Q0;
    }

    double r1 = Q1 - Q2 * std::cos(2.0 * M_PI * f);
    double i1 = Q2 * std::sin(2.0 * M_PI * f);

    double CN = std::cos((N - 1) * 2.0 * M_PI * f);
    double SN = std::sin((N - 1) * 2.0 * M_PI * f);

    double real = r1 * CN + i1 * SN;
    double imag = i1 * CN - r1 * SN;
    double amp = 2.0 * std::sqrt(real * real + imag * imag);
    double phase = M_PI / 2 + std::atan2(imag, real);
    double minoffset = testcycle_length * ((M_PI + M_PI / 2) / (2 * M_PI) - phase / (2.0 * M_PI));

    double residual_t = std::fmod(N, testcycle_length);
    double argument = 2 * M_PI * f * residual_t + phase;

    double maxoffset;
    if (argument <= M_PI / 2)
        maxoffset = std::round((M_PI / 2 - argument) * testcycle_length / (2 * M_PI));
    else if (argument <= 5 * M_PI / 2)
        maxoffset = std::round((5 * M_PI / 2 - argument) * testcycle_length / (2 * M_PI));
    else
        maxoffset = std::round((9 * M_PI / 2 - argument) * testcycle_length / (2 * M_PI));

    double minoffset2;
    if (argument < 3 * M_PI / 2)
        minoffset2 = std::round((3 * M_PI / 2 - argument) * testcycle_length / (2 * M_PI));
    else
        minoffset2 = std::round((7 * M_PI / 2 - argument) * testcycle_length / (2 * M_PI));

    return py::make_tuple(amp, phase, minoffset, minoffset2, maxoffset);
}

PYBIND11_MODULE(goertzel, m) {
    m.doc() = "Goertzel transform module with NumPy 2.0 support (via pybind11)";
    m.def("goertzel_general_shortened", &goertzel_general_shortened, "General Goertzel from vector and frequency indexes");
    m.def("goertzel_DFT", &goertzel_DFT, py::arg("testdata"), py::arg("testcycle_length"), py::arg("debug") = false);
}
