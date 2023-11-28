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
                           int k, double &richardson_clarification,
                           double precision);
double target_func(double x);


int main() {
    double a = 1;
    double b = 2;
    int starting_n = 2;
    double precision = 1e-6;
    
    double rich_clarif_rectangle;
    double rich_clarif_trapezoid;
    double rich_clarif_simpson;

    cout << "Function 2 * x + 1 / x" << endl << endl;
    cout << "Real solution: " << endl;
    cout << setprecision(13) << 3 + log(2) << endl << endl;
 
    cout << "Mean Rectangle Integral Method" << endl;
    double rectangle_integral = ApproximateIntegral(a, b, starting_n,
                                                    &target_func,
                                                    &MeanRectanglesMethod,
                                                    2, rich_clarif_rectangle,
                                                    precision);
    cout << "Integral (mean rectangle method): ";
    cout << setprecision(13) << rectangle_integral << endl;
    cout << "Richardson clarification: ";
    cout << setprecision(13) << rich_clarif_rectangle << endl << endl;
    

    cout << "Trapezoid Method" << endl;
    double trapezoid_integral = ApproximateIntegral(a, b, starting_n,
                                                    &target_func,
                                                    &TrapezoidalMethod,
                                                    2, rich_clarif_trapezoid,
                                                    precision);
    cout << "Integral (trapezoidal method): ";
    cout << setprecision(13) << trapezoid_integral << endl;
    cout << "Richardson clarification: ";
    cout << setprecision(13) << rich_clarif_trapezoid << endl << endl;
    
    cout << "Simpson Method" << endl;
    double simpson_integral = ApproximateIntegral(a, b, starting_n,
                                                  &target_func,
                                                  &SimpsonMethod,
                                                  4, rich_clarif_simpson,
                                                  precision);

    cout << "Integral (simpson method): ";
    cout << setprecision(13) << simpson_integral << endl;
    cout << "Richardson clarification: ";
    cout << setprecision(13) << rich_clarif_simpson << endl;

    return 0;
}


double target_func(double x) {
    return 2 * x + 1 / x;
}


double MeanRectanglesMethod(double x_start, double x_end, int n,
                            double (*func)(double)) {
    double h = (x_end - x_start) / n;

    cout << "Step: " << setprecision(5) << h << endl;

    double func_sum = 0;
    for (int i = 1; i <= n; ++i) {
        func_sum += func(x_start + (i - 0.5) * h);
    }
    return func_sum * h;
}


double TrapezoidalMethod(double x_start, double x_end, int n,
                         double (*func)(double)) {
    double h = (x_end - x_start) / n;

    cout << "Step: " << setprecision(5) << h << endl;

    double func_sum = 0.5 * (func(x_start) + func(x_end));
    for (int i = 1; i <= n - 1; ++i) {
        func_sum += func(x_start + i * h);
    }
    return func_sum * h;
}


double SimpsonMethod(double x_start, double x_end, int n,
                     double (*func)(double)) {
    double h = (x_end - x_start) / (2 * n);

    cout << "Step: " << setprecision(5) << h << endl;

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
                           int k, double& richardson_clarification,
                           double precision = 1e-6) {
    double curr_integral = integration_method(x_start, x_end,
                                              starting_n, func);
    double prev_integral;
    double absolute_error; 
    int n = 2 * starting_n;
    do {
        prev_integral = curr_integral;
        curr_integral = integration_method(x_start, x_end, n, func);
        n *= 2;
        absolute_error = abs(curr_integral - prev_integral);
    } while (absolute_error / (pow(2, k) - 1) >= precision);

    richardson_clarification = curr_integral \
        + (curr_integral - prev_integral) / (pow(2, k) - 1);

    return curr_integral;
}


