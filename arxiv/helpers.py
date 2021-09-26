from scipy import sparse
import numpy as np


class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0] - 1, self.n_columns]

        return sparse.csr_matrix((data, indices, indptr), shape=shape)
