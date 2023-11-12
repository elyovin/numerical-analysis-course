import sys
sys.path.append('../')

import copy
import numpy as np

from module_1.matrix_utils import Matrix


class GaussianElimination:
    def __init__(self) -> None:
        self.solution = None

    def solve(self, A: Matrix, b: Matrix) -> Matrix:
        assert_msg = 'Number of rows in A and b must be equal.'
        assert A.n_rows == b.n_rows, assert_msg

        assert_msg = 'b must be 1 column vector.'
        assert b.n_cols == 1, assert_msg

        # Create augmented matrix
        C = copy.deepcopy(A)
        C.add_column(b)

        # Forward pass
        n = A.n_rows
        for i in range(n):
            # Find maximum by modulus element and its index in j_th column
            a_max = C[i, i]
            idx_max = i
            for j in range(i + 1, n):
                if abs(C[j, i]) > abs(a_max):
                    a_max = C[j, i]
                    idx_max = j

            # Swap i_th row with row that contains maximum by modulus element
            for j in range(i, n + 1):
                C[i, j], C[idx_max, j] = C[idx_max, j], C[i, j]

            # Normalize each element in i_th row by first element in this row
            normalizer = C[i, i]
            for j in range(i, n + 1):
                C[i, j] = C[i, j] / normalizer

            # Substract i_th row from each row below
            for k in range(i + 1, n):
                multiplier = C[k, i]
                for j in range(i, n + 1):
                    C[k, j] = C[k, j] - C[i, j] * multiplier

        # Backward pass
        solution = Matrix(n, 1)
        for i in range(n - 1, -1, -1):
            tmp_sum = 0
            for j in range(i + 1, n):
                tmp_sum += solution[j, 0] * C[i, j]
            solution[i, 0] = C[i, n] - tmp_sum

        self.solution = solution
        return self.solution

    def print_solution(self) -> None:
        # generator = (f'x{i+1}={val}' for i, val in enumerate(self.solution))
        print('Solution:')
        for i in range(self.solution.n_rows):
            print(f'x{i+1}={self.solution[i, 0]}')

    def print_some_solution(self, low: int, high: int) -> None:
        '''
        generator = (
            f'x{low+i}={val}' for i, val in enumerate(self.solution[low-1:high-1])
        )
        '''
        assert_msg = 'Restriction: 0 <= low < high < solution.n_rows.'
        assert low >= 0 and high < self.solution.n_rows, assert_msg

        print('Solution:')
        for i in range(low, high):
            print(f'x{i+1}={self.solution[i, 0]}')


def task_1() -> None:
    A = Matrix(3, 3)
    A.enter([[1, 3.001, -3], [3, 3, 3], [-3, 3, 2]])
    b = Matrix(3, 1)
    b.enter([[2], [-6], [-16]])

    solver = GaussianElimination()
    solver.solve(A, b)
    print('Matrix 3x3')
    solver.print_solution()


def task_2() -> None:
    n = 1024
    A = Matrix(n, n)
    A.generate_random_uniform()

    solution = Matrix(n, 1)
    solution.enter([[i] for i in range(1, n + 1)])
    b = A * solution

    solver = GaussianElimination()
    solver.solve(A, b)
    print('Matrix 1024x1024')
    solver.print_some_solution(100, 120)


def task_3() -> None:
    A = np.array([[1, 3, -3], [3, 3, 3], [-3, 3, 2]])
    print('True condition number: ', np.linalg.cond(A))


def task() -> None:
    task_1()
    print()
    task_2()
    print()
    task_3()


if __name__ == '__main__':
    task()
