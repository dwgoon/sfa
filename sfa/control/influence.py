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
                      sparse=False,
                      dtype=None,
                      check_every=8,
                      use_tf32=True):
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
    dtype : numpy.dtype or str, optional
        Compute dtype for the native CUDA / CPU backends.
        Supported: ``numpy.float32``, ``numpy.float64``, ``numpy.float16``
        (CUDA only). When ``None`` (default), W's dtype is honored for CPU
        and inferred for CUDA. The cupy ``device='gpu:N'`` path keeps its
        historical float32 behavior.
    check_every : int, optional
        Native CUDA only. Number of iterations between host syncs (and the
        size of each CUDA Graph batch). Default 8. Set to 1 to disable
        graph capture / sync every iteration.
    use_tf32 : bool, optional
        Native CUDA only. Enable TF32 Tensor Core math mode for the FP32
        path. Default True. Ignored for FP64 / FP16.

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

    if max_iter < 1:
        raise ValueError("max_iter should be a positive integer.")

    device = device.lower()

    if 'cpu' in device:
        if sparse:
            ret = _compute_influence_cpu_sparse(W, alpha, beta, S,
                                               max_iter, tol, get_iter)
        else:
            ret = _compute_influence_cpu(W, alpha, beta, S,
                                        max_iter, tol, get_iter,
                                        dtype=dtype)
    elif 'cuda' in device:
        # Native CUDA path (sm_89-tuned: cuBLAS TF32 SGEMM + fused
        # add-identity/diff-norm kernel + CUDA Graph replay). Falls back
        # to the cupy or numpy paths if the extension is unavailable.
        if ':' in device:
            _, id_device = device.split(':')
        else:
            id_device = '0'
        ret = _compute_influence_cuda_native(W, alpha, beta, S,
                                             max_iter, tol, get_iter,
                                             int(id_device),
                                             check_every=check_every,
                                             use_tf32=use_tf32,
                                             dtype=dtype)
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


def _cpu_dtype(W, dtype):
    """Pick a sensible floating dtype for the CPU path.

    Honors the caller's ``dtype`` if given; otherwise upgrades integer W to
    float64 (numpy default) so iterative updates aren't truncated, and keeps
    floating W in its own dtype.
    """
    if dtype is not None:
        np_dt = np.dtype(dtype)
        if np_dt.kind != 'f':
            raise ValueError(f"CPU dtype must be floating; got {np_dt!r}")
        return np_dt
    w_dt = np.dtype(getattr(W, "dtype", np.float64))
    if w_dt.kind == 'f':
        return w_dt
    return np.dtype(np.float64)


def _compute_influence_cpu(W, alpha=0.5, beta=0.5, S=None,
                           max_iter=1000, tol=1e-6, get_iter=False,
                           dtype=None):
    """CPU influence backend.

    Two regimes:

    1. If the caller does not pass an initial ``S`` and does not need the
       iteration count, we solve the closed form
       ``S = beta * (I - alpha * W)^{-1}`` directly with LAPACK via
       ``scipy.linalg.solve``. This is a single getrf+getrs and is roughly
       an order of magnitude cheaper than the fixed-point iteration for
       typical convergence (~13 iters of N*N DGEMM at N=4096).

    2. Otherwise (caller-supplied S0, or get_iter=True) we keep the original
       fixed-point iteration so the historical convergence path / iteration
       count are preserved.
    """
    np_dt = _cpu_dtype(W, dtype)
    W = np.ascontiguousarray(W, dtype=np_dt)
    N = W.shape[0]

    # ---- Fast closed-form path ---------------------------------------------
    if S is None and not get_iter:
        # Single LAPACK getrf+getrs: solve  (I - alpha*W) X = beta*I.
        # alpha/beta arrive as Python double; cast to W's dtype so numpy
        # doesn't upcast the working matrices to float64 when W is float32.
        alpha_t = np_dt.type(alpha)
        beta_t  = np_dt.type(beta)
        A = np.eye(N, dtype=np_dt) - alpha_t * W
        B = beta_t * np.eye(N, dtype=np_dt)
        # overwrite_a/b avoid extra copies; check_finite=False skips an O(N^2)
        # NaN scan that is irrelevant for well-formed inputs.
        return sp.linalg.solve(A, B,
                               overwrite_a=True, overwrite_b=True,
                               check_finite=False)

    # ---- Iterative path (preserves get_iter semantics) ---------------------
    if S is not None:
        S1 = np.asarray(S, dtype=np_dt).copy()
    else:
        S1 = np.eye(N, dtype=np_dt)

    I = np.eye(N, dtype=np_dt)
    S2 = np.empty((N, N), dtype=np_dt)
    aW = alpha * W
    cnt = 0
    for cnt in range(1, max_iter + 1):
        np.dot(S1, aW, out=S2)
        S2 += I
        norm = np.linalg.norm(S2 - S1)
        if norm < tol:
            break
        S1[:, :] = S2

    S_fin = beta * S2
    if get_iter:
        return S_fin, cnt
    return S_fin


def _compute_influence_cpu_sparse(W, alpha, beta, S,
                                  max_iter, tol, get_iter):
    N = W.shape[0]
    if S is not None:
        S1 = sp.sparse.lil_matrix(S, dtype=np.float64)
    else:
        S1 = sp.sparse.lil_matrix(sp.sparse.eye(N, dtype=np.float64))


    I = sp.sparse.eye(N, dtype=np.float64)
    S2 = sp.sparse.lil_matrix((N, N), dtype=np.float64)
    aW = sp.sparse.csc_matrix(W, dtype=np.float64) * alpha
    cnt = 0
    for cnt in range(1, max_iter + 1):
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

    cnt = 0
    for cnt in range(1, max_iter + 1):
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


_NATIVE_DTYPE_MAP = {
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
    np.dtype(np.float16): "float16",
}


def _resolve_native_dtype(W, dtype):
    """Pick the on-device compute dtype.

    If ``dtype`` is given, it wins. Otherwise we honor ``W.dtype`` when it is
    one of the supported native dtypes; integer / unsupported dtypes default
    to float32 because that is what the native kernel was designed for.
    """
    if dtype is not None:
        np_dt = np.dtype(dtype)
    else:
        np_dt = np.dtype(getattr(W, "dtype", np.float32))
        if np_dt not in _NATIVE_DTYPE_MAP:
            np_dt = np.dtype(np.float32)
    if np_dt not in _NATIVE_DTYPE_MAP:
        raise ValueError(
            f"Unsupported native CUDA dtype {np_dt!r}; "
            "supported: float32, float64, float16.")
    return np_dt, _NATIVE_DTYPE_MAP[np_dt]


def _compute_influence_cuda_native(W, alpha, beta, S,
                                   max_iter, tol, get_iter,
                                   id_device=0,
                                   check_every=8,
                                   use_tf32=True,
                                   dtype=None):
    """Native CUDA backend: cuBLAS GEMM (TF32/FP32/FP64/FP16) + fused
    add-I/diff-norm kernel + CUDA Graph replay.

    Falls back to the cupy path if the native extension was not built,
    and to the numpy CPU path if cupy is also unavailable, so callers
    that rely on ``device='cuda:0'`` keep working in any environment.
    """
    if S is not None:
        # The native kernel currently starts from S(0) = I. If the caller
        # passes a custom initial S we route them through the cupy path so
        # behavior is preserved.
        return _compute_influence_gpu(W, alpha, beta, S,
                                      max_iter, tol, get_iter, id_device)

    from sfa import _cuda as _sfa_cuda
    if not _sfa_cuda.HAS_NATIVE:
        try:
            import cupy  # noqa: F401
        except ImportError:
            return _compute_influence_cpu(W, alpha, beta, S,
                                          max_iter, tol, get_iter)
        return _compute_influence_gpu(W, alpha, beta, S,
                                      max_iter, tol, get_iter, id_device)

    np_dt, dtype_str = _resolve_native_dtype(W, dtype)
    native = _sfa_cuda.native_module()
    W_cast = np.ascontiguousarray(W, dtype=np_dt)
    # NOTE: alpha/beta/tol are forwarded as Python float (= IEEE 754 double)
    # so the FP64 path keeps full precision; pybind11 binds them to C++
    # `double` and the FP32/FP16 paths cast down only inside the cuBLAS call.
    S_fin, num_iter = native.compute_influence_cuda(
        W_cast,
        alpha=alpha,
        beta=beta,
        max_iter=int(max_iter),
        tol=tol,
        device_id=int(id_device),
        check_every=int(check_every),
        use_tf32=bool(use_tf32),
        dtype=dtype_str,
    )

    if get_iter:
        return S_fin, num_iter
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
    """Prioritize target candidates.

    Parameters
    ----------
    df_splo : pandas.Series or pandas.DataFrame
        SPLO values for the candidate sources. Typically the column of
        ``sfa.splo(...)`` corresponding to ``output``.
    df_inf : pandas.DataFrame
        Influence values; rows are candidate sources, columns are
        outputs. ``df_inf[output]`` must be selectable.
    output : str
        Output node name (must be a column of ``df_inf``).
    dac : int
        Direction of activity change (DAC) sign filter applied to the
        influence on ``output``. Pass ``+1`` to keep sources with
        positive influence (inhibition suppresses the output) or ``-1``
        to keep sources with negative influence (activation suppresses
        the output).
    thr_rank : int or float, optional
        Top-rank threshold per SPLO bucket. An integer keeps the top-N
        rows of each bucket; a float in $(0, 1)$ keeps the top fraction.
        Default is ``3``.
    min_group_size : int, optional
        Minimum number of rows in a SPLO bucket for it to contribute
        candidates. Default is ``0`` (no filter).
    min_splo, max_splo : float, optional
        Restrict the candidate pool to SPLO values within this inclusive
        range. ``None`` (default) lets the helper infer the bounds.
    thr_inf : float, optional
        Influence-magnitude floor; rows whose ``|df_inf[output]|`` is
        less than or equal to this are dropped. Default is ``1e-10``.

    Returns
    -------
    targets : list of str
        Source-node names selected as control-target candidates.
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