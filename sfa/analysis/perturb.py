# -*- coding: utf-8 -*-

import numpy as np


def analyze_perturb(alg, data, targets, b=None, get_trj=False):
    """Perform signal flow analysis under perturbations.

    Parameters
    ----------
    alg : sfa.Algorithm
        Algorithm object.

    data : sfa.Data
        Data object which has perturbation data.

    targets : list
        List of node names, which are the keys of data.n2i.

    b : numpy.ndarray
        Basic vector for signaling sources or basal activities.

    get_trj : bool (optional)
        Decide to get the trajectory of activity change.
        
    Returns
    -------
    act : numpy.ndarray
        Change in the activities. It is usually calculated
        as x2 - x1, where x is
        the a vector of activities at steady-state.

    F : numpy.ndarray
        A matrix of signal flows.        
        It is usually calculated as W2*x1 - W1*x1,
        where W is weight matrix and
        x is a vector of activities at steady-state. 

    trj : numpy.ndarray (optional)
        Trajectory of activity change, which is returned
        if get_trj is True.
    """
    N = data.A.shape[0]

    if b is None:
        b = np.zeros((N,), dtype=np.float)
    elif b.size != N:
        raise TypeError("The size of b should be equal to %d"%(N))

    inds = []
    vals = []
    alg.apply_inputs(inds, vals)
    b[inds] = vals

    W_ctrl = alg.W.copy()
    x_ctrl, trj_ctrl = alg.propagate_iterative(
                                W_ctrl,
                                b,
                                b,
                                alg.params.alpha,
                                get_trj=get_trj)

    if data.has_link_perturb:
        W_pert = W_ctrl.copy()
        alg.apply_perturbations(targets, inds, vals, W_pert)
        alg.W = W_pert
    else:
        W_pert = W_ctrl
        alg.apply_perturbations(targets, inds, vals)

    b[inds] = vals
    x_pert, trj_pert = alg.propagate_iterative(
                                W_pert,
                                b,
                                b,
                                alg.params.alpha,
                                get_trj=get_trj)

    act_change = x_pert - x_ctrl

    if data.has_link_perturb:
        F = W_pert*x_pert - W_ctrl*x_ctrl
    else:
        F = W_ctrl*act_change

    ret = [act_change, F]  # return objects
    if get_trj:
        if trj_pert.shape[0] != trj_ctrl.shape[0]:
            trj_ctrl, trj_pert = resize_trj(trj_ctrl, trj_pert)
    
        trj_change = trj_pert - trj_ctrl
        ret.append(trj_change)

    return tuple(ret)


def resize_trj(trj_ctrl, trj_pert):
    # Prepare the comparison
    trjs = [trj_pert, trj_ctrl]
    ind_trjs = [0, 1]
    func_key = lambda x: trjs[x].shape[0]

    # Find smaller and bigger arrays.
    ind_smaller = min(ind_trjs, key=func_key)
    ind_bigger = max(ind_trjs, key=func_key)
    smaller = trjs[ind_smaller]
    bigger = trjs[ind_bigger]

    # Resize the smaller one.
    smaller_resized = np.zeros_like(bigger)
    smaller_resized[:smaller.shape[0], :] = smaller
    smaller_resized[smaller.shape[0]:, :] = smaller[-1, :]

    if ind_smaller == 0:
        trj_pert = smaller_resized
    elif ind_smaller == 1:
        trj_ctrl = smaller_resized
    else:
        err_msg = "Invalid index for trajectories: %d" % (ind_smaller)
        raise IndexError(err_msg)

    return trj_ctrl, trj_pert