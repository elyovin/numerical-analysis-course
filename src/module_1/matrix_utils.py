import random
from typing_extensions import Self


class Matrix:
    def __init__(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._matrix = [
            [0 for _ in range(self.n_cols)] for _ in range(self.n_rows)
        ]

    def enter_by_number(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self[i, j] = float(input())

        return self._matrix

    def add_column(self, column: Self) -> None:
        """
        Add column to the right of the matrix
        """
        assert_msg = f'Number of rows must be equal {self.n_rows}'
        assert column.n_rows == self.n_rows, assert_msg

        # assert_msg = f'Position index must be in [0, num_of_rows]'
        # assert 0 <= pos <= self.n_cols, assert_msg

        for row_1, row_2 in zip(self._matrix, column._matrix):
            row_1.extend(row_2)
        self.n_cols += 1

    def add_row(self, row: Self) -> None:
        """
        Add row to the bottom of the matrix
        """
        assert_msg = f'Number of columns must be equal {self.n_cols}'
        assert len(row) == self.n_cols, assert_msg

        # assert_msg = f'Position index must be in [0, num_of_columns]'
        # assert 0 <= pos <= self.n_rows, assert_msg

        self._matrix.extend(row._matrix)
        self.n_rows += 1

    def enter_by_user(self) -> Self:
        for i in range(self.n_rows):
            input_nums = list(map(float, input().split()))

            assert_msg = f'Number of columns must be equal {self.n_cols}'
            assert len(input_nums) == self.n_cols, assert_msg

            self[i] = input_nums

        return self

    def enter(self, matrix: list[list]) -> Self:
        assert_msg = f'Number of rows must be equal {self.n_rows}'
        assert len(matrix) == self.n_rows, assert_msg
        if len(matrix):
            assert_msg = f'Number of columns must be equal {self.n_cols}'
            assert len(matrix[0]) == self.n_cols, assert_msg

        self._matrix = matrix
        return self

    def generate_random_uniform(self, low: float = 0, high: float = 1) -> Self:
        """
        Fill matrix with uniformly distributed numbers
        from the closed interval [low, high]
        """

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self[i, j] = random.uniform(low, high)

        return self

    def generate_identity(self) -> Self:
        n = min(self.n_rows, self.n_cols)
        for i in range(n):
            self[i, i] = 1

        return self

    def transpose(self) -> Self:
        transposed_matrix = Matrix(self.n_cols, self.n_rows)
        transposed_matrix.enter([
            [0 for _ in range(self.n_rows)] for _ in range(self.n_cols)
        ])
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                transposed_matrix[j, i] = self[i, j]

        return transposed_matrix

    def norm(self, metrics: str = 'l2') -> float:
        assert_msg = 'Number of columns or rows must be 1'
        assert self.n_cols == 1 or self.n_rows == 1, assert_msg

        if metrics == 'l2':
            sum_ = 0
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    sum_ += self[i, j] ** 2
            result = sum_ ** 0.5
        elif metrics == 'l1':
            sum_ = 0
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    sum_ += abs(self[i, j])
            result = sum_
        elif metrics == 'inf':
            max_ = -float('inf')
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    if max_ < self[i, j]:
                        max_ = self[i, j]
            result = max_

        return result

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        assert 1 <= len(indices) <= 2, 'Number of indices must be equal 2'

        if len(indices) == 1:
            result = Matrix(1, self.n_cols)
            result._matrix = [self._matrix[indices[0]]]
        else:
            result = self._matrix[indices[0]][indices[1]]
        return result

    def __setitem__(self, indices, elem):
        if not isinstance(indices, tuple):
            indices = tuple(indices)

        assert 1 <= len(indices) <= 2, 'Number of indices must be equal 2'

        if len(indices) == 1:
            if isinstance(elem, (int, float)):
                for i in range(self.n_cols):
                    self._matrix[indices[0]][i] = elem
            elif isinstance(elem, list):
                assert_msg = f'Number of columns must be equal {self.n_cols}'
                assert len(elem) == self.n_cols, assert_msg

                self._matrix[indices[0]] = elem
        else:
            self._matrix[indices[0]][indices[1]] = elem

    def __sub__(self, other: Self) -> Self:
        """
        Matrix substraction
        """

        assert_msg = 'Shape of matrices must be equal.'
        assert (self.n_rows == other.n_rows
                and self.n_cols == other.n_cols), assert_msg

        new_matrix = Matrix(self.n_rows, self.n_cols)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                new_matrix[i, j] = self[i, j] - other[i, j]

        return new_matrix

    def __add__(self, other: Self) -> Self:
        """
        Matrix addition
        """

        assert_msg = 'Shape of matrices must be equal.'
        assert (self.n_rows == other.n_rows
                and self.n_cols == other.n_cols), assert_msg

        new_matrix = Matrix(self.n_rows, self.n_cols)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                new_matrix[i, j] = self[i, j] + other[i, j]

        return new_matrix

    def __mul__(self, other: Self | float | int) -> Self:
        """
        Matrix multiplication
        """
        if isinstance(other, (float, int)):
            new_matrix = Matrix(self.n_rows, self.n_cols)
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    new_matrix[i, j] = other * self[i, j]
        else:
            assert_msg = (
                'First matrix must have the same number of columns'
                'as the second matrix has rows.'
            )
            assert self.n_cols == other.n_rows, assert_msg

            new_matrix = Matrix(self.n_rows, other.n_cols)
            for i in range(self.n_rows):
                for j in range(other.n_cols):
                    new_matrix[i, j] = (
                        sum(
                            self[i, k]
                            * other[k, j] for k in range(self.n_cols))
                    )

        return new_matrix

    def __rmul__(self, other: Self | float | int) -> Self:
        return self * other

    def __str__(self) -> str:
        strings = [str(row) for row in self._matrix]
        return '\n'.join(strings)

    def __len__(self) -> int:
        return self.n_rows


if __name__ == '__main__':
    matrix_1 = [[1, 3, 1], [4, 1, 4], [2, 5, 2]]
    matrix_2 = [[1, 8], [2, 5], [3, 5]]
    m_1 = Matrix(3, 3)
    m_2 = Matrix(3, 2)
    m_1.enter(matrix_1)
    m_2.enter(matrix_2)
    print(m_1)

    print()
    m_1.transpose()
    print(m_1)

    print()
    print(m_2)
    m_2.transpose()
    print()
    print(m_2)
