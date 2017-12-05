import numpy as np


def calc_influence(W, alpha=0.5, E=None,
                   max_iter=100, tol=1e-6, get_iter=False):
    """Calculate the effect matrix.
       It estimates the effects of a node to the other nodes,
       by calculating partial derivative with respect to source nodes,
       based on a simple iterative method.

       Based on the below difference equation,

       x(t+1) = alpha*W.dot(x(t)) + (1-alpha)*b

       The effect matrix, E, is calculated using chain rule of
       partial derivative as follows.

       \begin{align}
        E_{ij} &= \frac{\partial{x_i}}{\partial{x_j}} \\
               &= (\alpha W + \alpha^2 W^2 +  ... + \alpha^{\infty}W^{\infty})_{ij} \\
               &\approx (\alpha W + \alpha^2 W^2 +  ... + \alpha^{l}W^{l})_{ij} \\
       \end{align}

       This is the summation of the weight multiplications along all paths
       including cycles. $E_{ij}$ denotes the effect of node (j) on node (i).

       An iterative method for an approximated solution is as follows.

        E(t+1) = \alpha WE(t) + \alpha W,

       where $E(0) = I$ and $E(1) = \alpha W$ $(t>1)$.

       The iteration continues until $||E(t+1) - E(t)|| \leq tol$.


    Parameters
    ----------
    W : numpy.ndarray
        Weight matrix.
    alpha : int, optional
        Hyperparameter for propagation rate.
    E : numpy.ndarray, optional
        Initial effect matrix.
    max_iter : int, optional
        The maximum iteration number for the estimation.
    tol : float, optional
        Tolerance for terminating the iteration.
    get_iter : bool, optional
        Determine whether the actual iteration number is returned.

    Returns
    -------
    E : numpy.ndarray
        Effect matrix.
    num_iter : int, optional
        The actual number of iteration.
    """
    # TODO: Test rendering the above mathematical expressions in LaTeX form.

    N = W.shape[0]
    if E is not None:
        E1 = E
    else:
        E1 = np.eye(N, dtype=np.float)

    aW = alpha * W
    for cnt in range(max_iter):
        E2 = E1.dot(aW) + aW
        if np.linalg.norm(np.abs(E2 - E1)) < tol:
            break
        # end of if
        E1 = E2
    # end of for
    if get_iter:
        return E2, cnt
    else:
        return E2