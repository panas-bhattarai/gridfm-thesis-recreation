"""Admittance-weighted random-walk positional encoding (GridFM v0.2).
"""
import torch

K_STEPS = 16   


def transition_matrix(edge_index, edge_attr, num_nodes):
    """Dense admittance-weighted random-walk matrix M
    """
    w = torch.sqrt(edge_attr[:, 0] ** 2 + edge_attr[:, 1] ** 2)   # |y| per edge
    W = torch.zeros(num_nodes, num_nodes, dtype=edge_attr.dtype)
    W.index_put_((edge_index[0], edge_index[1]), w, accumulate=True)
    deg = W.sum(dim=1, keepdim=True)                              # weighted degree
    # every bus in our data is connected (datagen discards islanded samples),
    # but guard against a zero row rather than divide by zero
    return W / deg.clamp(min=1e-30)


def rwpe(edge_index, edge_attr, num_nodes, k=K_STEPS):
    """Random-walk positional encoding, shape (num_nodes, k)
    """
    M = transition_matrix(edge_index, edge_attr, num_nodes)
    out = torch.empty(num_nodes, k, dtype=M.dtype)
    P = M.clone()
    out[:, 0] = P.diagonal()
    for t in range(1, k):
        P = P @ M
        out[:, t] = P.diagonal()
    return out


def add_rwpe(graphs, k=K_STEPS):
    """Attach g.rwpe (float32, (n, k)) to every PyG Data object in the list.

    Cheap enough (~ms per graph) to run at load time; deterministic, so no
    need to persist alongside the .pt files.
    """
    for g in graphs:
        g.rwpe = rwpe(g.edge_index, g.edge_attr, g.num_nodes, k=k).to(torch.float32)
    return graphs


def concat_rwpe(graphs, k=K_STEPS):
    """add_rwpe, then concatenate the encoding onto g.x (n, 9) -> (n, 9+k).
    """
    add_rwpe(graphs, k)
    for g in graphs:
        assert g.x.shape[1] == 9, "concat_rwpe applied twice or on non-standard x"
        g.x = torch.cat([g.x, g.rwpe], dim=1)
    return graphs
