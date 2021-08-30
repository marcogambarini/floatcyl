#! /usr/bin/env python3

def vector_index(n, m, Nn, Nm):
    """Computes the transformation from indices (m,n) to the single index
    of interaction vectors and matrices
    """
    return (n + Nn)*(Nm + 1) + m

def inverse_vector_indices(ii, Nn, Nm):
    """Computes the transformation from the single index
    of interaction vectors and matrices to indices (m,n)
    """
    n, m = divmod((ii - Nn*(Nm+1)),(Nm+1))

    return n, m
