def assert_matrix(X, Y):
    assert X.shape[0] == X.shape[0]
    assert Y.shape[1] == Y.shape[1]

    row_num = X.shape[0]
    col_num = Y.shape[1]
    for row in range(row_num):
        for col in range(col_num):
            assert X[row][col] == Y[row][col]
