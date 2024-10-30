import networkx as nx
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def projection_order1(D, max = 1e+10):
    d = D.size()[0]
    row_norms = torch.linalg.norm(D, ord = 2, dim = 1)**2
    col_norms = torch.linalg.norm(D, ord = 2, dim = 0)**2
    assert torch.max(row_norms) < max and torch.max(col_norms) < max
    order = torch.zeros(d, dtype = torch.int) - 1
    first_index = 0
    last_index = 0
    for i in range(d):
        min_row_norm, row_index = torch.min(row_norms,dim=0)
        min_col_norm, col_index = torch.min(col_norms,dim=0)
        if min_row_norm <= min_col_norm:
            remove_index = row_index.item()
            order[-1 - last_index] = row_index + 0
            last_index += 1
        else:
            remove_index = col_index.item()
            order[first_index] = col_index + 0
            first_index += 1

        row_norms -= D[:,remove_index].square()
        row_norms[remove_index] = max
        col_norms -= D[remove_index].square()
        col_norms[remove_index] = max
    return order

def mask_from_order(order, main_mask):
    d = order.size()[0]
    mask = torch.zeros_like(main_mask) != 0
    for i, o in enumerate(order):
        for j in range(d-i-1):
            mask[o,order[j+i+1]] = True
    return mask


def is_dag(B):
    """
    Check whether B corresponds to a DAG.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))

def threshold_till_dag(B):
    """
    Remove the edges with smallest absolute weight until a DAG is obtained.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] weighted matrix.

    Return
    ------
    B: numpy.ndarray
        [d, d] weighted matrix of DAG.
    dag_thres: float
        Minimum threshold to obtain DAG.
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    B1 = np.copy(B*0)
   
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]),reverse=True)

    for weight, j, i in sorted_weight_indices_ls:
        B1[j,i] = B[j,i] + 0.
       
        if is_dag(B1)==False:
        # Remove the edge that does not give DAG
            B1[j, i] = 0
            dag_thres = abs(weight)

    return B1, dag_thres


def postprocess(B, graph_thres=0.):
    """
    Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] weighted matrix.
    graph_thres: float
        Threshold for weighted matrix. Default: 0.0.

    Return
    ------
    B: numpy.ndarray
        [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)
    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B,_= threshold_till_dag(B)

    return B

