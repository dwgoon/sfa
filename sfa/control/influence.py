import numpy as np
import scipy as sp
import pandas as pd


def calc_influence(W,
                   alpha=0.9,
                   beta=0.1,
                   S=None,
                   rtype='df',
                   outputs=None,
                   n2i=None,
                   max_iter=1000,
                   tol=1e-6,
                   get_iter=False,
                   device="cpu",
                   sparse=False):
    """Calculate the influence matrix.
       It estimates the effects of a node to the other nodes,
       by calculating partial derivative with respect to source nodes,
       based on a simple iterative method.

       Based on the below difference equation,

       x(t+1) = alpha*W.dot(x(t)) + (1-alpha)*b

       The influence matrix, S, is calculated using chain rule of
       partial derivative as follows.

       \begin{align}
        S_{ij} &= \frac{\partial{x_i}}{\partial{x_j}} \\
               &= (I + \alpha W + \alpha^2 W^2 +  ... + \alpha^{\infty}W^{\infty})_{ij} \\
               &\approx (I + \alpha W + \alpha^2 W^2 +  ... + \alpha^{l}W^{l})_{ij} \\
       \end{align}

       This is the summation of the weight multiplications along all paths
       including cycles. $S_{ij}$ denotes the influence of node (j) on node (i).

       An iterative method for an approximated solution is as follows.

        S(t+1) = \alpha WS(t) + I,

       where $S(0) = \beta I$ and $S(1) = \beta(I + \alpha W)$ $(t>1)$.

       The iteration continues until $||S(t+1) - S(t)|| \leq tol$.


    Parameters
    ----------
    W : numpy.ndarray
        Weight matrix.
    alpha : float, optional
        Hyperparameter for adjusting the effect of signal flow.
    beta : float, optional
        Hyperparameter for adjusting the effect of basal activity.
    S : numpy.ndarray, optional
        Initial influence matrix.
    rtype: str (optional)
        Return object type: 'df' or 'array'.
    outputs: list (or iterable) of str, optional
        Names of output nodes, which is necessary for 'df' rtype.
    n2i: dict, optional
        Name to index dict, which is necessary for 'df' rtype.
    max_iter : int, optional
        The maximum iteration number for the estimation.
    tol : float, optional
        Tolerance for terminating the iteration.
    get_iter : bool, optional
        Determine whether the actual iteration number is returned.
    device : str, optional, {'CPU', 'GPU:0', 'GPU:1', ...}
        Select which device to use. 'CPU' is default.
    sparse : bool, optional
        Use sparse matrices for the computation.

    Returns
    -------
    S : numpy.ndarray, optional
        2D array of influence.
    df : pd.DataFrame, optional
        Influences for each output in DataFrame.
    num_iter : int, optional
        The actual number of iteration.
    """
    # TODO: Test rendering the above mathematical expressions in LaTeX form.

    if max_iter < 2:
        raise ValueError("max_iter should be greater than 2.")

    device = device.lower()

    if 'cpu' in device:
        if sparse:
            ret = _calc_influence_cpu_sparse(W, alpha, beta, S,
                                               max_iter, tol, get_iter)
        else:
            ret = _calc_influence_cpu(W, alpha, beta, S,
                                        max_iter, tol, get_iter)
    elif 'gpu'in device:
        _, id_device = device.split(':')
        ret = _calc_influence_gpu(W, alpha, beta, S,
                                  max_iter, tol, get_iter, id_device)

    if get_iter:
        S_ret, num_iter = ret
    else:
        S_ret = ret

    if rtype == 'array':
        return ret
    elif rtype == 'df':
        if not outputs:
            err_msg = "outputs should be designated for 'df' return type."
            raise ValueError(err_msg)

        df = pd.DataFrame(columns=outputs)

        for trg in outputs:
            for src in n2i:
                if src == trg:
                    df.loc[src, trg] = np.inf

                idx_src = n2i[src]
                idx_trg = n2i[trg]
                df.loc[src, trg] = S_ret[idx_trg, idx_src]

        if get_iter:
            return df, num_iter
        else:
            return df
    else:
        raise ValueError("Unknown return type: %s"%(rtype))


def _calc_influence_cpu(W, alpha=0.5, beta=0.5, S=None,
                        max_iter=1000, tol=1e-6, get_iter=False):
    N = W.shape[0]
    if S is not None:
        S1 = S
    else:
        S1 = np.eye(N, dtype=np.float)

    I = np.eye(N, dtype=np.float)
    S2 = np.zeros_like(W)
    aW = alpha * W
    for cnt in range(max_iter):
        S2[:, :] = S1.dot(aW) + I
        norm = np.linalg.norm(S2 - S1)
        if norm < tol:
            break
        # end of if
        S1[:, :] = S2
    # end of for

    S_fin = beta * S2
    if get_iter:
        return S_fin, cnt
    else:
        return S_fin


def _calc_influence_cpu_sparse(W, alpha, beta, S,
                               max_iter, tol, get_iter):
    N = W.shape[0]
    if S is not None:
        S1 = S
    else:
        S1 = sp.sparse.lil_matrix(sp.sparse.eye(N, dtype=np.float))


    I = sp.sparse.eye(N, dtype=np.float)
    S2 = sp.sparse.lil_matrix((N,N), dtype=np.float)
    aW = sp.sparse.csc_matrix(alpha * W)
    for cnt in range(max_iter):
        S2[:, :] = S1.dot(aW) + I
        norm = sp.sparse.linalg.norm(S2 - S1)
        if norm < tol:
            break
        # end of if
        S1[:, :] = S2
    # end of for

    S_fin = beta * S2
    if get_iter:
        return S_fin, cnt
    else:
        return S_fin


def _calc_influence_gpu(W, alpha=0.5, beta=0.5, S=None,
                        max_iter=1000, tol=1e-6, get_iter=False,
                        id_device=0):    
    import cupy as cp
    cp.cuda.Device(id_device).use()    
    N = W.shape[0]
    I = cp.eye(N, dtype=cp.float32) #np.eye(N, N, dtype=np.float)
    if S is not None:
        S1 = cp.array(S, dtype=cp.float32)
    else:
        S1 = cp.eye(N, dtype=cp.float32)
    
    S2 = cp.zeros((N,N), dtype=cp.float32)
    aW = alpha * cp.array(W, dtype=cp.float32)
    
    tol_gpu = cp.array(tol)
    
    for cnt in range(max_iter):
        S2[:, :] = cp.dot(S1, aW) + I
        mat_norm = cp.linalg.norm(S2 - S1)
        if mat_norm < tol_gpu:
            break
        # end of if
        S1[:, :] = S2
    # end of for
    
    S_fin = beta*S2
    if get_iter:
        return S_fin, cnt
    else:
        return S_fin