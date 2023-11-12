import sys
sys.path.append('../')

import numpy as np
from typing import Callable

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['font.size'] = 12

from module_1.thomas_algorithm import solve as tridiagonal_solve


class CubicSplineInterpolation:
    def __init__(self, f: Callable) -> None:
        self.func = f  # function to be interpolated

    def _get_c_coefs(self, n: int, h: float, y: list) -> list:
        right_side = [0] * (n + 1)
        right_side[0], right_side[-1] = 0, 0

        for i in range(1, n):
            right_side[i] = (y[i - 1] - 2 * y[i] + y[i + 1]) / h

        upper_diagonal = [h / 3 for _ in range(n)]
        upper_diagonal[-1] = 0
        
        main_diagonal = [4 * h / 3 for _ in range(n + 1)]
        main_diagonal[0] = 1
        main_diagonal[-1] = 1

        lower_diagonal = [h / 3 for _ in range(n)]
        lower_diagonal[0] = 0

        solution = tridiagonal_solve(
            main_diagonal, upper_diagonal,
            lower_diagonal, right_side
        )
        return solution

    def fit(self, x_start: float, x_end: float, n: int) -> None:
        self.x_start = x_start
        self.x_end = x_end
        self.n = n
        self.h = (self.x_end - self.x_start) / self.n
        x_values = [self.x_start + i * self.h for i in range(self.n + 1)]
        y_values = list(map(self.func, x_values))
        
        self.a = [0] * self.n
        self.b = [0] * self.n
        self.c = self._get_c_coefs(self.n, self.h, y_values)
        self.d = [0] * self.n

        for i in range(self.n):
            self.a[i] = y_values[i]
            self.b[i] = (
                (y_values[i + 1] - y_values[i]) / self.h
                - self.h * (2 * self.c[i] + self.c[i + 1]) / 3
            )
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3 * self.h)

    def predict(self, x: float) -> float:
        assert self.x_start <= x <= self.x_end, 'x must be within fitted range'
        
        if x == self.x_end:
            position = self.n - 1
        else:
            position = int(x / self.h)

        residual = x - position * self.h
        y_pred = (
            self.a[position]
            + self.b[position] * residual
            + self.c[position] * residual ** 2
            + self.d[position] * residual ** 3
        )

        return y_pred

    def visualize(self, title: str) -> None:
        # Get interpolation splines and original function values
        m = 50
        sub_h = (self.x_end - self.x_start) / (m * self.n)
        x_values = [0] * (m * self.n)
        y_pred_values = [0] * (m * self.n)
        y_target_values = [0] * (m * self.n)
        for i in range(self.n):
            plt.axvline(x=self.x_start + i * self.h, color='black', linewidth=0.3)
            for j in range(m):
                x = self.x_start + (i * m + j) * sub_h
                y_pred = self.predict(x)
                y_target = self.func(x)

                x_values[i * m + j] = x
                y_pred_values[i * m + j] = y_pred
                y_target_values[i * m + j] = y_target


        # Get function values in the middle of intervals
        x_values_mid = (
            [self.x_start + (i - 0.5) * self.h for i in range(1, self.n + 1)]
        )
        y_pred_values_mid = (
            list(map(self.predict, x_values_mid))
        )
        y_target_values_mid = (
            list(map(self.func, x_values_mid))
        )
        
        y_min = min(*y_pred_values_mid, *y_target_values_mid)
        y_max = max(*y_pred_values_mid, *y_target_values_mid)

        # Plot
        plt.axvline(x=self.x_end, color='black', linewidth=0.3)
        plt.plot(
            x_values, y_target_values, 
            label='original function', color='red'
        )
        plt.scatter(
            x_values_mid, y_target_values_mid, color='red'
        )
        
        plt.plot(
            x_values, y_pred_values, 
            label='spline interpolation', color='blue'
        )
        plt.scatter(
            x_values_mid, y_pred_values_mid, color='blue'
        )
        
        plt.xticks(
            [self.x_start + i * self.h for i in range(self.n)] + [self.x_end]
        )
        plt.legend()
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim([y_min - abs(y_min), y_max + abs(y_max)])
        plt.show()
        

def print_table(x_values: list, f: Callable, solver: CubicSplineInterpolation) -> None:
    y_values = list(map(f, x_values))
    y_pred_values = list(map(solver.predict, x_values))

    print()
    print(
        '{: <12}'.format('x_i'),
        '{: <12}'.format('f(x_i)'),
        '{: <12}'.format('y_i'),
        sep=' | '
    )
    for i in range(len(y_values)):
        print(
            f'{x_values[i]: <12.3f}',
            f'{y_pred_values[i]: <12.3f}',
            f'{y_values[i]: <12.3f}',
            sep=' | ',
        )
  

def task() -> None:
    def f(x: float) -> float:
        return np.tan(x)

    def identity(x: float) -> float:
        return x

    x_start, x_end = 0, 2
    n = 10
    h = (x_end - x_start) / n
    solver = CubicSplineInterpolation(f)
    solver.fit(x_start, x_end, n)
    solver.visualize('Интерполяция тангенса кубическими сплайнами')
    x_values = [x_start + (i - 0.5) * h for i in range(1, n + 1)]
    print_table(x_values, f, solver)

    solver_identity = CubicSplineInterpolation(identity)
    solver_identity.fit(x_start, x_end, n)
    solver_identity.visualize('Интерполяция линейной функции кубическими сплайнами')
    print_table(x_values, identity, solver_identity)
