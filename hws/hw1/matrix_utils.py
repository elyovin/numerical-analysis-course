#TODO: write manipulation with Matrix and Matrix

class Matrix:
    def __init__(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self._matrix = (
            [[None for j in range(self.n_cols)] for i in range(self.n_rows)]
        )

    def enter_by_number(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self._matrix[i][j] = float(input())

        return self._matrix

    def add_column(self, column: list, pos: int) -> None:
        assert_msg = f'Number of rows must be equal {self.n_rows}'
        assert len(column) == self.n_rows, assert_msg

        assert_msg = f'Position index must be in [0, num_of_rows]'
        assert 0 <= pos <= self.n_cols, assert_msg

        for i, row in enumerate(self._matrix):
            row.insert(pos, column[i])
        self.n_cols += 1

    def add_row(self, row: list, pos: int) -> None:
        assert_msg = f'Number of columns must be equal {self.n_cols}'
        assert len(row) == self.n_cols, assert_msg

        assert_msg = f'Position index must be in [0, num_of_columns]'
        assert 0 <= pos <= self.n_rows, assert_msg

        self._matrix.insert(pos, row)
        self.n_rows += 1

    def enter_by_user(self):
        for i in range(self.n_rows):
            input_nums = list(map(float, input().split()))
            
            assert_msg = f'Number of columns must be equal {self.n_cols}'
            assert len(input_nums) == self.n_cols, assert_msg

            self._matrix[i] = input_nums

        return self._matrix

    def enter(self, matrix: list[list]):
        assert_msg = f'Number of rows must be equal {self.n_rows}'
        assert len(matrix) == self.n_rows, assert_msg
        if len(matrix):
            assert_msg = f'Number of columns must be equal {self.n_cols}'
            assert len(matrix[0]) == self.n_cols, assert_msg

        self._matrix = matrix
        return self._matrix
        

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = tuple(indices)

        assert 1 <= len(indices) <= 2, 'Number of indices must be equal 2'

        if len(indices) == 1:
            result = self._matrix[indices[0]]
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


    def __str__(self) -> str:
        strings = [str(row) for row in self._matrix]
        return '\n'.join(strings)

    def __len__(self) -> int:
        return self.n_rows


if __name__ == '__main__':
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    m = Matrix(4, 3)
    m.enter(matrix)
    print(m)
    print()
    m.add_column([15, 16, 17, 18], 3)
    print(m)
    print()
    m.add_row([1, 1, 1, 1], 1)
    print(m)
    print()

    m[3, 2] = 777
    print(m)

