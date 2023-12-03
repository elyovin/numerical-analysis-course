#include <iostream>
#include <math.h>
#include <vector>
#include <iomanip>


void F(const std::vector<float>&, float, std::vector<float>&);
void F_solution(float, std::vector<float>&);
void const_multiply_vector(float, std::vector<float>&);


int main() {
    int m = 2;
    float a = 0;
    float b = 2;
    int n = 10;
    float h = (b - a) / n;  // step
    std::vector<std::vector<float>> u(
        n + 1,
        std::vector<float>(m)
    );

    // Initial conditions
    u[0][0] = 0;
    u[0][1] = -2;

    std::vector<float> K1(m);
    std::vector<float> K2(m);
    std::vector<float> K3(m);
    std::vector<float> K4(m);


    std::vector<float> K1_arg(m);
    std::vector<float> K2_arg(m);
    std::vector<float> K3_arg(m);
    std::vector<float> K4_arg(m);

    std::vector<float> tmp_calc(m);
    for (int i = 0; i < n; ++i) {
        float t = a + i * h;

        // K1 calculation
        for (int j = 0; j < m; ++j) {
            K1_arg[j] = u[i][j];
        }
        F(K1_arg, t, tmp_calc);
        const_multiply_vector(h, tmp_calc);
        std::copy(std::begin(tmp_calc), std::end(tmp_calc), std::begin(K1));

        // K2 calculation
        for (int j = 0; j < m; ++j) {
            K2_arg[j] = u[i][j] + K1[j] / 2;
        }
        F(K2_arg, t + h / 2, tmp_calc);
        const_multiply_vector(h, tmp_calc);
        std::copy(std::begin(tmp_calc), std::end(tmp_calc), std::begin(K2));

        // K3 calculation
        for (int j = 0; j < m; ++j) {
            K3_arg[j] = u[i][j] + K2[j] / 2;
        }
        F(K3_arg, t + h / 2, tmp_calc);
        const_multiply_vector(h, tmp_calc);
        std::copy(std::begin(tmp_calc), std::end(tmp_calc), std::begin(K3));

        // K4 calculation
        for (int j = 0; j < m; ++j) {
            K4_arg[j] = u[i][j] + K3[j];
        }
        F(K4_arg, t + h, tmp_calc);
        const_multiply_vector(h, tmp_calc);
        std::copy(std::begin(tmp_calc), std::end(tmp_calc), std::begin(K4));

        for (int j = 0; j < m; ++j) {
            u[i + 1][j] = u[i][j] +  (K1[j] + 2 * K2[j]
                                      + 2 * K3[j] + K4[j]) / 6;
        }
    }

    std::vector<float> analytic_solution(m);
    std::cout << "  t  | x1_analytic | x1_numeric | x2_analytic | x2_numeric"
        << std::endl;
    for (int i = 0; i < n + 1; ++i) {
        float t = a + i * h;
        F_solution(t, analytic_solution);
        std::cout << std::setw(4) << t << " |" 
            << std::setw(12) << analytic_solution[0] << " |"
            << std::setw(11) << u[i][0] << " |"
            << std::setw(12) << analytic_solution[1] << " |"
            << std::setw(11) << u[i][1] << std::endl;
    }

    return 0;
}


void F(const std::vector<float>& x, float t, std::vector<float>& values) {
    values[0] = 3 * x[0] + 2 * x[1] + 3 * exp(2 * t);
    values[1] = x[0] + 2 * x[1] + exp(2 * t);
}


void F_solution(float t, std::vector<float>& values) {
    values[0] = exp(t) - exp(2 * t);
    values[1] = -exp(t) - exp(2 * t);
}


void const_multiply_vector(float constant, std::vector<float>& values) {
    for (int i = 0; i < values.size(); i++) {
        values[i] *= constant;
    }

}
