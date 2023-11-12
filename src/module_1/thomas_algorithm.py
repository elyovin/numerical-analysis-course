import sys
sys.path.append('../')

import numpy as np


def solve(main_diagonal: list, upper_diagonal: list,
          lower_diagonal: list, b: list) -> list:
    n: int = len(b)  # number of equations
    alphas: list = [0 for _ in range(n - 1)]
    betas: list = [0 for _ in range(n - 1)]
    solution: list = [0 for _ in range(n)]

    alphas[0] = -upper_diagonal[0]
    betas[0] = b[0]

    # Forward pass
    for i in range(1, n - 1):
        alphas[i] = (
            -upper_diagonal[i]
            / (alphas[i - 1] * lower_diagonal[i - 1] + main_diagonal[i])
        )

        betas[i] = (
            (b[i] - lower_diagonal[i - 1] * betas[i - 1])
            / (lower_diagonal[i - 1] * alphas[i - 1] + main_diagonal[i])
        )

    # Backward pass
    solution[-1] = (
        (-lower_diagonal[-1] * betas[-1] + b[-1])
        / (1 + lower_diagonal[-1] * alphas[-1])
    )
    for i in range(n - 2, -1, -1):
        solution[i] = alphas[i] * solution[i + 1] + betas[i]

    return solution


def print_solution(solution: list) -> None:
    generator = (
        f'x{i+1}={val}' for i, val in enumerate(solution)
    )
    print(*generator, sep='\n')


def task() -> None:
    main_diagonal = [1, -2, 0, 1]
    upper_diagonal = [9, -8, 3]
    lower_diagonal = [-1, -9, -8]
    b = [33, -76, -42, -69]

    print('My solution: ')
    solution = solve(main_diagonal, upper_diagonal, lower_diagonal, b)
    print_solution(solution)

    print('True solution: ')
    matrix = np.array([[1, 9, 0, 0], [-1, -2, -8, 0],
                       [0, -9, 0, 3], [0, 0, -8, 1]])
    solution = np.linalg.solve(matrix, b)
    print_solution(solution)
