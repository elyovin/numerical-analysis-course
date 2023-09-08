# TODO: write tests

import copy

from matrix_utils import Matrix


class GaussianElimination:
    def __init__(self) -> None:
        self.solution = []

    def solve(self, A: Matrix, b: list) -> list:
        assert_msg = 'Number of rows in A and b must be equal.'
        assert A.n_rows == len(b), assert_msg_dims

        '''
        assert_msg = 'b must be 1 column vector.'
        assert == 1, assert_msg
        '''

        # Create augmented matrix
        C = copy.deepcopy(A)
        n = A.n_rows
        C.add_column(b, C.n_cols)

        ''' Forward pass '''
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

        ''' Backward pass '''
        solution = [None for _ in range(n)]
        for i in range(n - 1, -1, -1):
            sum_ = 0
            for j in range(i + 1, n):
                sum_ += solution[j] * C[i, j]
            solution[i] = C[i, n] - sum_

        self.solution = solution
        return self.solution

    def print_solution(self):
        generator = (f'x{i+1}={val}' for i, val in enumerate(self.solution))
        print('Solution:', *generator, sep='\n')
            

def enter_data() -> tuple[Matrix, list]:
    n = int(input('Enter n: '))
    print('Enter A:')
    m = Matrix(n, n)
    m.enter_by_user()
    print('Enter b:')
    b = list(map(int, input().split()))

    return m, b


if __name__ == '__main__':
    # m = Matrix(3, 3)
    # m.enter([[1, 3, 2], [5, 7, 3], [5, 2, 1]])
    # m.add_column([2, -6, -16], m.n_cols)
    A, b = enter_data()
    solver = GaussianElimination()
    solver.solve(A, b)
    solver.print_solution()
    

