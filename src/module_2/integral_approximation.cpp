#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cmath>
using namespace std;


double MeanRectanglesMethod(double x_start, double x_end, int n,
                            double (*func)(double));
double TrapezoidalMethod(double x_start, double x_end, int n,
                         double (*func)(double));
double SimpsonMethod(double x_start, double x_end, int n,
                     double (*func)(double));
double ApproximateIntegral(double x_start, double x_end, int starting_n,
                           double (*func)(double),
                           double (*integration_method)(double, double,
                                                        int, double(double)),
                           int k, double precision);
double target_func(double x);


int main() {
    double a = 1;
    double b = 2;
    int starting_n = 2;
    double precision = 1e-6;

    cout << std::setprecision(10) << endl;    
    double rectangle_integral = ApproximateIntegral(a, b, starting_n,
                                                    &target_func,
                                                    &MeanRectanglesMethod,
                                                    2, precision);

    double trapezoid_integral = ApproximateIntegral(a, b, starting_n,
                                                    &target_func,
                                                    &TrapezoidalMethod,
                                                    2, precision);

    double simpson_integral = ApproximateIntegral(a, b, starting_n,
                                                  &target_func,
                                                  &SimpsonMethod,
                                                  4, precision);



    cout << "Function 2 * x + 1 / x" << endl;
    cout << "Integral (mean rectangle method): " << rectangle_integral << endl;
    cout << "Integral (trapezoidal method): " << trapezoid_integral << endl;
    cout << "Integral (simpson method): " << simpson_integral << endl;
    return 0;
}


double target_func(double x) {
    return 2 * x + 1 / x;
}


double MeanRectanglesMethod(double x_start, double x_end, int n,
                            double (*func)(double)) {
    double h = (x_end - x_start) / n;
    double func_sum = 0;
    for (int i = 1; i <= n; ++i) {
        func_sum += func(x_start + (i - 0.5) * h);
    }
    return func_sum * h;
}


double TrapezoidalMethod(double x_start, double x_end, int n,
                         double (*func)(double)) {
    double h = (x_end - x_start) / n;
    double func_sum = 0.5 * (func(x_start) + func(x_end));
    for (int i = 1; i <= n - 1; ++i) {
        func_sum += func(x_start + i * h);
    }
    return func_sum * h;
}


double SimpsonMethod(double x_start, double x_end, int n,
                     double (*func)(double)) {
    double h = (x_end - x_start) / (2 * n);
    double func_sum = func(x_start) + func(x_end);
    for (int i = 1; i <= 2 * n - 1; ++i) {
        int coef = (i % 2 + 1) * 2;
        func_sum += coef * func(x_start + i * h);
    }
    return h / 3 * func_sum;
}


double ApproximateIntegral(double x_start, double x_end, int starting_n,
                           double (*func)(double),
                           double (*integration_method)(double, double,
                                                        int, double(double)),
                           int k, double precision = 1e-6) {
    double curr_integral = integration_method(x_start, x_end,
                                              starting_n, func);
    double absolute_error; 
    int n = 2 * starting_n;
    do {
        double prev_integral = curr_integral;
        curr_integral = integration_method(x_start, x_end, n, func);
        n *= 2;
        absolute_error = abs(curr_integral - prev_integral);
    } while (absolute_error / (pow(2, k) - 1) >= precision);

    return curr_integral;
}


