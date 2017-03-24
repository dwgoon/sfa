# -*- coding: utf-8 -*-

import numpy as np

def analyze_perturb(alg, data, targets, get_trj=False):
    """Perform signal flow analysis under perturbations.

    Parameters
    ----------
    alg : sfa.Algorithm
        Algorithm object.

    data : sfa.Data
        Data object which has perturbation data.

    targets : list
        List of node names, which are the keys of data.n2i.

    get_trj : bool (optional)
        Decide to get the trajectory of activity change.

    Returns
    -------
    act : 1-dim numpy.ndarray
        Change in the activities. It is usually calculated
        as x_perturb - x_control, where x is
        the steady-state of activities.

    trj : 2-dim numpy.ndarray (optional)
        Trajectory of activity change, which is returned
        if get_trj is True.
    """
    N = data.A.shape[0]
    b = np.zeros((N,), dtype=np.float)

    inds = []
    vals = []
    alg.apply_inputs(inds, vals)
    b[inds] = vals
    x_ctrl, trj_ctrl = alg.propagate_iterative(
                                alg.W,
                                b,
                                b,
                                alg.params.alpha,
                                get_trj=get_trj)

    alg.apply_perturbations(targets, inds, vals)
    b[inds] = vals
    x_pert, trj_pert = alg.propagate_iterative(
                                alg.W,
                                b,
                                b,
                                alg.params.alpha,
                                get_trj=get_trj)

    act_change = x_pert - x_ctrl
    if get_trj:
        trj_change = trj_pert - trj_ctrl
        return act_change, trj_change
    else:
        return act_change

