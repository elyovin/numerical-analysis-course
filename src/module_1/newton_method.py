from matrix_utils import Matrix
from gaussian_elimination_method import GaussianElimination


def f(vector: Matrix) -> Matrix:
    x, y = vector[0, 0], vector[1, 0]
    matrix = Matrix(2, 1)
    matrix.enter(
        [
            [x ** 4 + 6 * (x ** 2) * (y ** 2) + y ** 4 - 136],
            [(x ** 3) * y + (y ** 3) * x - 30]
        ]
    )
    return matrix


def jacobian(vector: Matrix) -> Matrix:
    x, y = vector[0, 0], vector[1, 0]
    matrix = Matrix(2, 2)
    matrix.enter(
        [
            [
                4 * (x ** 3) + 12 * x * (y ** 2),
                12 * (x ** 2) * y + 4 * (y ** 3)
            ],
            [
                3 * (x ** 2) * y + y ** 3,
                x ** 3 + 3 * x * (y ** 2)
            ]
        ]
    )

    return matrix


def solve(initial_approximation: Matrix,
          epsilon: float = 1e-10) -> tuple[Matrix, int]:
    solver = GaussianElimination()
    x = Matrix(2, 1)
    x_prev = initial_approximation
    n_iterations = 0
    while True:
        x = x_prev + solver.solve(jacobian(x_prev), (-1) * f(x_prev))
        if (x - x_prev).norm() < epsilon:
            break
        x_prev = x
        n_iterations += 1

    return x, n_iterations


def task() -> None:
    x_0 = Matrix(2, 1)
    x_0.enter([[0.5], [1.8]])
    epsilon = 1e-10
    solution, n_iterations = solve(x_0, epsilon)
    print(f'Initial approximation: x_0={x_0[0, 0]}, y_0={x_0[1, 0]}')
    print(f'Epsilon: {epsilon}')
    print(f'Number of iterations: {n_iterations}')
    print(f'Solution: x={solution[0, 0]} y={solution[1, 0]}')
