import numpy as np


def calc_influence(W, alpha=0.5, beta=0.5, S=None,
                   max_iter=1000, tol=1e-6, get_iter=False):
    """Calculate the influence matrix.
       It estimates the effects of a node to the other nodes,
       by calculating partial derivative with respect to source nodes,
       based on a simple iterative method.

       Based on the below difference equation,

       x(t+1) = alpha*W.dot(x(t)) + (1-alpha)*b

       The influence matrix, S, is calculated using chain rule of
       partial derivative as follows.

       \begin{align}
        E_{ij} &= \frac{\partial{x_i}}{\partial{x_j}} \\
               &= (\alpha W + \alpha^2 W^2 +  ... + \alpha^{\infty}W^{\infty})_{ij} \\
               &\approx (\alpha W + \alpha^2 W^2 +  ... + \alpha^{l}W^{l})_{ij} \\
       \end{align}

       This is the summation of the weight multiplications along all paths
       including cycles. $S_{ij}$ denotes the influence of node (j) on node (i).

       An iterative method for an approximated solution is as follows.

        S(t+1) = \alpha WS(t) + \alpha W,

       where $S(0) = I$ and $S(1) = \alpha W$ $(t>1)$.

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
    max_iter : int, optional
        The maximum iteration number for the estimation.
    tol : float, optional
        Tolerance for terminating the iteration.
    get_iter : bool, optional
        Determine whether the actual iteration number is returned.

    Returns
    -------
    S : numpy.ndarray
        Influence matrix.
    num_iter : int, optional
        The actual number of iteration.
    """
    # TODO: Test rendering the above mathematical expressions in LaTeX form.

    N = W.shape[0]
    I = np.eye(N, N, dtype=np.float)
    if S is not None:
        S1 = S
    else:
        S1 = np.eye(N, dtype=np.float)

    aW = alpha * W
    for cnt in range(max_iter):
        S2 = S1.dot(aW) + I
        if np.linalg.norm(np.abs(S2 - S1)) < tol:
            break
        # end of if
        S1 = S2
    # end of for
    if get_iter:
        return beta*S2, cnt
    else:
        return beta*S2