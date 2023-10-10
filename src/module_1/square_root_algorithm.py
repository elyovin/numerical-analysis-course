import math
import numpy as np

from matrix_utils import Matrix


class SquareRoot:
    def __init__(self) -> None:
        self.solution = None

    def solve(self, A: Matrix, b: Matrix) -> Matrix:
        assert_msg = 'Number of rows in A and b must be equal.'
        assert A.n_rows == b.n_rows, assert_msg

        assert_msg = 'b must be 1 column vector.'
        assert b.n_cols == 1, assert_msg

        # Forward pass
        n = A.n_rows
        S = Matrix(n, n)
        D = Matrix(n, n)
        for i in range(n):
            tmp_sum = 0
            for k in range(i):
                tmp_sum += D[k, k] * S[k, i] ** 2
            tmp_sum = A[i, i] - tmp_sum

            D[i, i] = 1 if tmp_sum >= 0 else -1
            S[i, i] = abs(tmp_sum) ** 0.5

            for j in range(i + 1, n):
                tmp_sum = 0
                for k in range(i):
                    tmp_sum += D[k, k] * S[k, j] * S[k, i]
                tmp_sum = A[i, j] - tmp_sum
                S[i, j] = tmp_sum / (S[i, i] * D[i, i])

        S_T = S.transpose()
        X = Matrix(n, 1)
        X[0, 0] = b[0, 0] / S[0, 0]
        for i in range(1, n):
            tmp_sum = 0
            for k in range(i):
                tmp_sum += X[k, 0] * S_T[i, k]
            tmp_sum = b[i, 0] - tmp_sum
            X[i, 0] = tmp_sum / S_T[i, i]

        # Backward pass
        y = D * X
        solution = Matrix(n, 1)
        solution[n - 1, 0] = y[n - 1, 0] / S[n - 1, n - 1]
        for i in range(n - 1):
            k = n - i - 2
            tmp_sum = 0
            for j in range(k + 1, n):
                tmp_sum += solution[j, 0] * S[k, j]
            tmp_sum = y[k, 0] - tmp_sum
            solution[k, 0] = tmp_sum / S[k, k]

        self.solution = solution
        return solution

    def print_solution(self) -> None:
        # generator = (f'x{i+1}={val}' for i, val in enumerate(self.solution))
        print('Solution:')
        for i in range(self.solution.n_rows):
            print(f'x{i+1}={self.solution[i, 0]}')


def find_max_nondiag_elem(A: Matrix) -> tuple[float, int, int]:
    max_elem = -float('inf')
    n = A.n_rows
    p, q = 0, 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if abs(A[i, j]) > max_elem:
                max_elem = abs(A[i, j])
                p, q = i, j
    return max_elem, p, q


def jacobi_rotation(A: Matrix, epsilon: float = 1e-6) -> tuple[Matrix, Matrix]:
    n = A.n_rows
    max_elem, p, q = find_max_nondiag_elem(A)
    eigen_vectors = Matrix(n, n)
    eigen_vectors.generate_identity()

    while max_elem > epsilon:
        if abs(A[p, p] - A[p, q]) < 1e-6:
            phi = math.pi / 4
        else:
            phi = math.atan(2 * A[p, q] / (A[p, p] - A[q, q])) / 2

        # Rotation matrix
        U = Matrix(n, n)
        U.generate_identity()
        U[p, p] = math.cos(phi)
        U[p, q] = -math.sin(phi)
        U[q, p] = math.sin(phi)
        U[q, q] = math.cos(phi)

        A = U.transpose() * A * U
        eigen_vectors = eigen_vectors * U
        max_elem, p, q = find_max_nondiag_elem(A)

    eigen_values = Matrix(n, 1)
    for i in range(n):
        eigen_values[i, 0] = A[i, i]

    return eigen_values, eigen_vectors


def task() -> None:
    n = 4
    A = Matrix(n, n)
    b = Matrix(n, 1)
    matrix = [
        [8, -5, 4, 4],
        [-5, -1, 3, -7],
        [4, 3, -6, 8],
        [4, -7, 8, -3]
    ]

    A.enter(matrix)
    b.enter([[-161], [82], [-53], [-100]])

    solver = SquareRoot()
    solver.solve(A, b)
    solver.print_solution()
    my_eigen_values, my_eigen_vectors = jacobi_rotation(A)

    print('\nMy eigen values and eigen vectors:')
    for i in range(n):
        print(
            f'lambda_{i+1}={my_eigen_values[i, 0]}',
            f'v_{i+1}={my_eigen_vectors.transpose()[i]}\n',
            sep='\n'
        )

    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    print('\nTrue eigen values and eigen vectors')
    for i in range(n):
        print(
            f'lambda_{i+1}={eigen_values[i]}',
            f'v_{i+1}={eigen_vectors[:, i]}\n',
            sep='\n'
        )

    max_eigen, min_eigen = -float('inf'), float('inf')
    for i in range(len(eigen_values)):
        max_eigen = max(max_eigen, abs(my_eigen_values[i, 0]))
        min_eigen = min(min_eigen, abs(my_eigen_values[i, 0]))

    print(
        'My condition number:',
        max_eigen / min_eigen,
        'True condition number:',
        np.linalg.cond(matrix),
        sep='\n'
    )
