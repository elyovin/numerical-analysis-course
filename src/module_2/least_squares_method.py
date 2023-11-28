import sys
sys.path.append('../')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['font.size'] = 12

import numpy as np

from typing import Callable

from module_1.matrix_utils import Matrix
from module_1.gaussian_elimination_method import GaussianElimination


class LeastSquaresMethod:
    def __init__(self, degree: int) -> None:
        assert degree >= 0, 'Degree must be non negative number'
        self.degree = degree  # degree of polynomial
        self.solution = None  # matrix of coefficients of polynomial
        self.x_train = None
        self.y_train = None

    def fit(self, x: Matrix, y: Matrix) -> Matrix:
        self.x_train = x
        self.y_train = y
        
        error_msg = 'For each x there is need to be y'
        assert x.n_cols == y.n_cols and x.n_rows == y.n_rows, error_msg
        
        b = Matrix(self.degree + 1, self.degree + 1) 
        for p in range(self.degree + 1):
            for q in range(self.degree + 1):
                b[p, q] = sum(x[0, i] ** (p + q) for i in range(x.n_cols))

        c = Matrix(self.degree + 1, 1)
        for p in range(self.degree + 1):
            c[p, 0] = sum(y[0, i] * x[0, i] ** p for i in range(x.n_cols))

        solver = GaussianElimination()
        self.solution = solver.solve(b, c)
        return self.solution

    def predict_value(self, x: float) -> float:
        y = 0
        for i in range(self.solution.n_rows):
            y += self.solution[i, 0] * x ** i
        return y
        
    def predict_matrix(self, x: Matrix) -> Matrix:
        y = Matrix(x.n_rows, x.n_cols)
        for i in range(x.n_rows):
            for j in range(x.n_cols):
                y[i, j] = self.predict_value(x[i, j])
        return y

    def get_residuals(self, x: Matrix, y: Matrix) -> Matrix:
        y_predicted = self.predict_matrix(x)
        return y_predicted - y

    def visualize(self, title: str, n: int = 30) -> None:
        if self.x_train is not None and self.y_train is not None:
            x_min = min(self.x_train[0, i] for i in range(self.x_train.n_cols))
            x_max = max(self.x_train[0, i] for i in range(self.x_train.n_cols))
            x_values = np.linspace(x_min, x_max, n)
            y_values = list(map(self.predict_value, x_values))

            plt.scatter(
                [self.x_train[0, i] for i in range(self.x_train.n_cols)],
                [self.y_train[0, i] for i in range(self.y_train.n_cols)],
                color='blue',
                label='train points'
            )
            plt.plot(x_values, y_values, color='red', label='interpolation polynomial')
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(title)
            plt.show()


def tabulate(f: Callable, start: float, end: float,
             n: int = 10) -> tuple[Matrix, Matrix]:
    h = (end - start) / n
    
    x_values = Matrix(1, n + 1)
    for i in range(n + 1):
        x_values[0, i] = start + h * i

    y_values = Matrix(1, n + 1)
    for i in range(n + 1):
        y_values[0, i] = f(start + h * i)

    return x_values, y_values


def task() -> None:
    def f(x: float) -> float:
        return np.tanh(x)

    x_start, x_end = 0, 2
    x_values, y_values = tabulate(f, x_start, x_end)

    # Line
    ls_line = LeastSquaresMethod(1)
    ls_line.fit(x_values, y_values)
    ls_line.visualize('Интерполяция линейной функцией')
    residuals_line = ls_line.get_residuals(x_values, y_values)
    residuals_line_sum = sum(residuals_line[0, i] ** 2 for i in range(residuals_line.n_cols))

    # Square
    ls_square = LeastSquaresMethod(2)
    ls_square.fit(x_values, y_values)
    ls_square.visualize('Интерполяция квадратичной функцией')
    residuals_square = ls_square.get_residuals(x_values, y_values)
    residuals_square_sum = sum(residuals_square[0, i] ** 2 for i in range(residuals_square.n_cols))
    
    # Cube
    ls_cube = LeastSquaresMethod(3)
    ls_cube.fit(x_values, y_values)
    ls_cube.visualize('Интерполяция кубической функцией')
    residuals_cube = ls_cube.get_residuals(x_values, y_values)
    residuals_cube_sum = sum(residuals_cube[0, i] ** 2 for i in range(residuals_cube.n_cols))
    

    # Print table
    print(
        '{: <12}'.format('x_i'),
        '{: <12}'.format('y_i = f(x_i)'),
        '{: <12}'.format('phi_1(x_i)'),
        '{: <12}'.format('phi_2(x_i)'),
        '{: <12}'.format('phi_3(x_i)'),
        '{: <12}'.format('eps_1(x_i)'),
        '{: <12}'.format('eps_2(x_i)'),
        '{: <12}'.format('eps_3(x_i)'),
        sep=' | '
    )
    for i in range(x_values.n_cols):
        print(
            f'{x_values[0, i]: <12.3f}',
            f'{y_values[0, i]: <12.3f}',
            f'{ls_line.predict_value(x_values[0, i]): <12.3f}',
            f'{ls_square.predict_value(x_values[0, i]): <12.3f}',
            f'{ls_cube.predict_value(x_values[0, i]): <12.3f}',
            f'{residuals_line[0, i]: <12.3f}',
            f'{residuals_square[0, i]: <12.3f}',
            f'{residuals_cube[0, i]: <12.3f}',
            sep=' | ',
        )
    print(
        ' ' * 72,
        f'{residuals_line_sum: <12.3f}',
        f'{residuals_square_sum: <12.3f}',
        f'{residuals_cube_sum: <12.3f}',
        sep=' | '
    )
        

    
        
    