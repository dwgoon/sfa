# -*- coding: utf-8 -*-

import numpy as np

from .np import NetworkPropagation
from .np import NetworkPropagationParameterSet


def create_algorithm(abbr):
    return SignalPropagation(abbr)
# end of def


class SignalPropagationParameterSet(NetworkPropagationParameterSet):
    def initialize(self):
        super().initialize()
# end of class ParameterSet


class SignalPropagation(NetworkPropagation):

    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Signal propagation algorithm"
    # end of def __init__

    def prepare_exact_solution(self):
        r"""
        Prepare the matrix $M$ for the closed-form steady-state solution.

        Starting from

        $$
        x(t+1) = \alpha W x(t) + (1 - \alpha) b,
        $$

        as $t \to \infty$ both $x(t+1)$ and $x(t)$ converge to the
        stationary state $s$, giving

        $$
        s = \alpha W s + (1 - \alpha) b
          \implies (I - \alpha W) s = (1 - \alpha) b
          \implies s = M b,
          \quad M = (1 - \alpha)(I - \alpha W)^{-1}.
        $$

        This method computes and caches $M$.
        """
        W = self._W
        a = self._params.alpha
        M0 = np.eye(W.shape[0]) - a*W
        self._M = (1-a)*np.linalg.inv(M0)
        self._weight_matrix_invalidated = False
    # end of def _prepare_exact_solution

    def prepare_iterative_solution(self):
        pass  # Nothing...
    # end of def prepare_iterative_solution

    def propagate_exact(self, b):
        if self._weight_matrix_invalidated:
            self.prepare_exact_solution()
            
        return self._M.dot(b)
    # end of def propagate_exact

    def propagate_iterative(self,
                            W,
                            xi,
                            b,
                            a=0.5,
                            lim_iter=1000,
                            tol=1e-5,
                            get_trj=False,
                            device="cpu",
                            dtype=None,
                            check_every=8,
                            use_tf32=True):

        device = (device or "cpu").lower()
        if "cuda" in device:
            out = _propagate_iterative_cuda(
                W, xi, b, a, lim_iter, tol, device, dtype, get_trj,
                check_every, use_tf32)
            if out is not None:
                # (x, num_iter) when get_trj=False, (x, trajectory) when True.
                return out

        # ---- CPU path ------------------------------------------------------
        # Pick the working dtype. None means "honor xi.dtype if floating,
        # else upgrade to float64". This keeps backward compatibility with
        # callers that pass integer/float64 inputs.
        if dtype is not None:
            np_dt = np.dtype(dtype)
            if np_dt.kind != 'f':
                raise ValueError(f"dtype must be floating; got {np_dt!r}")
        else:
            xi_dt = np.dtype(getattr(xi, "dtype", np.float64))
            np_dt = xi_dt if xi_dt.kind == 'f' else np.dtype(np.float64)

        W_  = np.ascontiguousarray(W,  dtype=np_dt)
        xi_ = np.ascontiguousarray(xi, dtype=np_dt)
        b_  = np.ascontiguousarray(b,  dtype=np_dt)

        x_t1 = xi_.copy()
        x_t2 = np.empty_like(x_t1)
        one_minus_a = np_dt.type(1.0 - a)
        a_ = np_dt.type(a)

        # Pre-allocated trajectory buffer: shape (lim_iter + 1, N), so we can
        # write rows in place without growing a Python list. Rows are reused
        # if convergence is reached early; we slice to the actual count at
        # the end (no copy, np view).
        if get_trj:
            trj = np.empty((lim_iter + 1, x_t1.size), dtype=np_dt)
            trj[0] = x_t1

        num_iter = 0
        for i in range(lim_iter):
            # x_t2 = a*W @ x_t1 + (1-a)*b  (no temporary allocations)
            np.dot(W_, x_t1, out=x_t2)
            x_t2 *= a_
            x_t2 += one_minus_a * b_
            num_iter += 1
            if get_trj:
                trj[num_iter] = x_t2
            if np.linalg.norm(x_t2 - x_t1) <= tol:
                break
            # Swap views instead of copying: x_t1 becomes the freshest state,
            # and we recycle x_t1's old buffer as the next x_t2.
            x_t1, x_t2 = x_t2, x_t1

        if get_trj:
            # Slice to actual rows (no copy).
            return x_t2, trj[: num_iter + 1]
        return x_t2, num_iter

    # end of def propagate_iterative


_SP_NATIVE_DTYPE_MAP = {
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
    np.dtype(np.float16): "float16",
}


def _propagate_iterative_cuda(W, xi, b, alpha, lim_iter, tol,
                              device, dtype, get_trj,
                              check_every=8, use_tf32=True):
    """Dispatch to the native CUDA signal-propagation kernel.

    Returns ``(x, num_iter)`` (no trajectory) or ``(x, trajectory)`` on
    success, or ``None`` if the native extension is unavailable (the caller
    then falls back to numpy).
    """
    try:
        from sfa import _cuda as _sfa_cuda
    except ImportError:
        return None
    if not _sfa_cuda.HAS_NATIVE:
        return None

    id_device = int(device.split(":")[1]) if ":" in device else 0

    if dtype is not None:
        np_dt = np.dtype(dtype)
    else:
        np_dt = np.dtype(getattr(W, "dtype", np.float32))
        if np_dt not in _SP_NATIVE_DTYPE_MAP:
            np_dt = np.dtype(np.float32)
    if np_dt not in _SP_NATIVE_DTYPE_MAP:
        raise ValueError(
            f"Unsupported native CUDA dtype {np_dt!r}; "
            "supported: float32, float64, float16.")
    dtype_str = _SP_NATIVE_DTYPE_MAP[np_dt]

    W_c = np.ascontiguousarray(W,  dtype=np_dt)
    x_c = np.ascontiguousarray(xi, dtype=np_dt)
    b_c = np.ascontiguousarray(b,  dtype=np_dt)
    native = _sfa_cuda.native_module()
    # alpha/tol pass through as Python double; pybind binds them to C++ double
    # so the FP64 path keeps full precision and FP32/FP16 cast at the GEMV.
    x_fin, second = native.compute_signal_propagation_cuda(
        W_c, x_c, b_c,
        alpha=alpha,
        max_iter=int(lim_iter),
        tol=tol,
        device_id=id_device,
        check_every=int(check_every),
        use_tf32=bool(use_tf32),
        dtype=dtype_str,
        return_trajectory=bool(get_trj),
    )
    return x_fin, second  # second is num_iter or trajectory
# end of def class SignalPropagation
