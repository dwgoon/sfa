from collections import Counter

import numpy as np
import scipy as sp
import pandas as pd


def compute_influence(W,
                      alpha=0.9,
                      beta=0.1,
                      S=None,
                      rtype='df',
                      outputs=None,
                      n2i=None,
                      max_iter=1000,
                      tol=1e-7,
                      get_iter=False,
                      device="cpu",
                      sparse=False):
    r"""Compute the influence matrix.

    Estimates how each source node affects every other node,
    based on the signal flow propagation

    $$
    x(t+1) = \alpha W x(t) + \beta b.
    $$

    Using the chain rule of partial derivatives, the influence is

    $$
    S_{ij} = \frac{d x_i}{d b_j}
           = \beta \bigl(I + \alpha W + \alpha^2 W^2 + \cdots\bigr)_{ij},
    $$

    which converges to $S^* = \beta (I - \alpha W)^{-1}$ when the
    spectral radius of $\alpha W$ is less than one. $S_{ij}$ is the
    influence of node $j$ on node $i$.

    An iterative method (paper form) computes

    $$
    S(t+1) = \alpha W S(t) + \beta I, \qquad S(0) = \beta I,
    $$

    and stops when $\lVert S(t+1) - S(t) \rVert_F \le \mathrm{tol}$.
    The shipped implementation iterates as
    $S(t+1) = S(t)\,\alpha W + I$ with $S(0) = I$ and applies the
    $\beta$ factor at the end; both arrangements converge to the same
    $\beta (I - \alpha W)^{-1}$.

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
            ret = _compute_influence_cpu_sparse(W, alpha, beta, S,
                                               max_iter, tol, get_iter)
        else:
            ret = _compute_influence_cpu(W, alpha, beta, S,
                                        max_iter, tol, get_iter)
    elif 'gpu'in device:
        _, id_device = device.split(':')
        ret = _compute_influence_gpu(W, alpha, beta, S,
                                  max_iter, tol, get_iter, id_device)
        
        if rtype == 'df':
            import cupy as cp
            if isinstance(ret, cp.ndarray):
                ret = cp.asnumpy(ret)

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


def _compute_influence_cpu(W, alpha=0.5, beta=0.5, S=None,
                           max_iter=1000, tol=1e-6, get_iter=False):
    N = W.shape[0]
    if S is not None:
        S1 = S
    else:
        S1 = np.eye(N, dtype=np.float64)

    I = np.eye(N, dtype=np.float64)
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


def _compute_influence_cpu_sparse(W, alpha, beta, S,
                                  max_iter, tol, get_iter):
    N = W.shape[0]
    if S is not None:
        S1 = S
    else:
        S1 = sp.sparse.lil_matrix(sp.sparse.eye(N, dtype=np.float64))


    I = sp.sparse.eye(N, dtype=np.float64)
    S2 = sp.sparse.lil_matrix((N,N), dtype=np.float64)
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


def _compute_influence_gpu(W, alpha=0.5, beta=0.5, S=None,
                           max_iter=1000, tol=1e-6, get_iter=False,
                           id_device=0):    
    import cupy as cp
    cp.cuda.Device(id_device).use()    
    N = W.shape[0]
    I = cp.eye(N, dtype=cp.float32) #np.eye(N, N, dtype=np.float64)
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



def arrange_si(
        df_splo,
        df_inf,
        output,
        min_splo=None,
        max_splo=None,
        thr_inf=1e-10,
        ascending=True):

    # SPLO-Influence data
    if not min_splo:
        min_splo = df_splo.min()

    if not max_splo:
        max_splo = df_splo.max()

    mask_splo = (min_splo <= df_splo) & (df_splo <= max_splo)
    df_splo = df_splo[mask_splo]

    df_splo = pd.DataFrame(df_splo)
    df_splo.columns = ['SPLO']

    if output in df_splo.index:
        df_splo.drop(output, inplace=True)

    index_common = df_splo.index.intersection(df_inf.index)
    df_inf = pd.DataFrame(df_inf.loc[index_common])

    mark_drop = df_inf[output].abs() <= thr_inf
    df_inf.drop(df_inf.loc[mark_drop, output].index,
                inplace=True)


    df_si = df_inf.join(df_splo.loc[index_common])
    df_si.index.name = 'Source'
    df_si.reset_index(inplace=True)

    cnt_splo = Counter(df_si['SPLO'])
    splos = sorted(cnt_splo.keys())

    si = {}
    for i, splo in enumerate(splos):
        df_sub = df_si[df_si['SPLO'] == splo]
        df_sub = df_sub.sort_values(by=output,
                                    ascending=ascending)
        #num_items = df_sub[output].count()
        #influence = np.zeros((cnt_max,))  # Influence
        #num_empty = cnt_max - num_items
        #influence[num_empty:] = df_sub[output]
        #names = df_sub['Source'].tolist()
        si[splo] = df_sub  #[output]

    return si


def prioritize(df_splo,
               df_inf,
               output,
               dac,
               thr_rank=3,
               min_group_size=0,
               min_splo=None,
               max_splo=None,
               thr_inf=1e-10,
):
    """Prioritize target candiates.
    
    Parameters
    ----------
    df_splo : pandas.DataFrame
        Dataframe for SPLO information.
    df_inf : pandas.DataFrame
        Dataframe for influence information.
    output : str
        Names of output node, which is necessary for 'df_inf'.
    dac : int
        Direction of activity change (DAC) of the output.
    thr_rank : int or float
        Rank to filter out the entities.
        The entities whose ranks are greater than thr_rank survive.
    min_group_size : int
        Minimum group size to be satisfied.
    """
    ascending = True if dac < 0 else False

    df_inf_dac = df_inf[np.sign(df_inf[output]) == dac]
    si = arrange_si(df_splo,
                    df_inf_dac,
                    output,
                    min_splo, 
                    max_splo,
                    thr_inf,
                    ascending)
    targets = []
    for splo in si:
        # Get the group of this SPLO.
        df_sub = si[splo]  
       
        if df_sub.shape[0] < min_group_size:
           continue       
       
        # Get the entities that have the designated dac.
        df_sub = df_sub[np.sign(df_sub[output]) == dac]
        
        # Get the enetities whose rank exceeds the threshods.
        if 0 < thr_rank < 1:
            ix_max_rank = int(thr_rank * df_sub.shape[0])
            if ix_max_rank == 0:
                ix_max_rank = df_sub.shape[0] 
        else:
            ix_max_rank = thr_rank
        
        #print(ix_max_rank)
        df_top = df_sub.iloc[:ix_max_rank, :]

        targets.extend(df_top['Source'].tolist())
    # end of for
    return targets