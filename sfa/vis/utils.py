
import numpy as np
import networkx as nx


__all__ = [
    'compute_graphics',
]


def _rgb_to_hex(tup):
    """Convert RGBA to #AARRGGBB
    """
    tup = tuple(tup)
    if len(tup) == 3:
        return '#%02x%02x%02x' % tup
    elif len(tup) == 4:
        return '#%02x%02x%02x%02x' % (tup[3], tup[0], tup[1], tup[2])
    else:
        raise ValueError("Array or tuple for RGB or RGBA should be given.")


def compute_graphics(
        F,
        act,
        A,
        n2i,
        lw_min=1.0,
        lw_max=10.0,
        pct_link=90,
        pct_act=50,
        dg=None):
    """Compute graphics of signal flow.

    This method performs a calculation for generating colors
    of nodes and links for visualizing purpose.

    Parameters
    ----------
    F : numpy.ndarray
        A matrix of signal flows.
        It is usually calculated as W2*x1 - W1*x1,
        where W is weight matrix and
        x is a vector of activities at steady-state.

    act : numpy.ndarray
        Change in the activities. It is usually calculated
        as x2 - x1, where x is
        the a vector of activities at steady-state.

    A : numpy.ndarray
        Adjacency matrix of the network.

    n2i : dict
        Name to index dictionary.

    lw_min : float, optional
        Minimum link width, which is also used for unchanged flow.

    lw_max : float, optional
        Maximum link width.

    pct_link : int, optional
        Percentile of link width, which is used to set
        the maximum value for setting link widths.
        Default value is 90.

    pct_act : int, optional
        Percentile of activity, which is used to set
        the maximum value for coloring nodes.
        Default value is 50.

    dg : NetworkX.DiGraph, optional
        Existing NetworkX object to contain graphics information
        for visualizing nodes and links.

    Returns
    -------
    dg : NetworkX.DiGraph
        NetworkX object containing graphics information
        for visualizing nodes and links.

    """

    if not dg:
        dg = nx.DiGraph()
        dg.add_nodes_from(n2i)

    _compute_graphics_nodes(dg, n2i, act, pct_act)
    _compute_graphics_links(dg, n2i, A, F, pct_link, lw_min, lw_max)

    return dg


def _compute_graphics_nodes(dg, n2i, act, pct_act):
    color_white = np.array([255, 255, 255])
    color_up = np.array([255, 0, 0])
    color_dn = np.array([0, 0, 255])

    abs_act = np.abs(act)
    thr = np.percentile(abs_act, pct_act)
    thr = 1 if thr == 0 else thr

    arr_t = np.zeros_like(act)
    for i, elem in enumerate(act):
        t = np.clip(np.abs(elem) / thr, a_min=0, a_max=1)
        arr_t[i] = t

    for iden, idx in n2i.items():
        fold = act[idx]

        if fold > 0:
            color = color_white + arr_t[idx] * (color_up - color_white)
        elif fold <= 0:
            color = color_white + arr_t[idx] * (color_dn - color_white)

        color = _rgb_to_hex(np.int32(color))

        data = dg.nodes[iden]
        data['FILL_COLOR'] = color
        data['BORDER_WIDTH'] = 2
        data['BORDER_COLOR'] = _rgb_to_hex((40, 40, 40))
    # end of for


def _compute_graphics_links(dg, n2i, A, F, pct_link, lw_min, lw_max):
    i2n = {val: key for key, val in n2i.items()}

    log_flows = np.log10(np.abs(F[F.to_numpy().nonzero()]))
    flow_max = log_flows.max()
    flow_min = log_flows.min()
    flow_thr = np.percentile(log_flows, pct_link)

    ir, ic = A.to_numpy().nonzero()  # F.to_numpy().nonzero()
    for i, j in zip(ir, ic):
        tgt = i2n[i]
        src = i2n[j]
        f = F[i, j]

        #link = net.nxdg[src][tgt]['VIS']
        dg.add_edge(src, tgt)
        data = dg.edges[src, tgt]


        #header_old = link.header
        #args_header = header_old.width, header_old.height, header_old.offset
        if f > 0:
            sign_link = +1 # PosHeader(*args_header)
            color_link = _rgb_to_hex((255, 10, 10, 70))
        elif f < 0:
            sign_link = -1  # NegHeader(*args_header)
            color_link = _rgb_to_hex((10, 10, 255, 70))
        else:  # When flow is zero, show the sign of the original link.
            if A[i, j] > 0:
                sign_link = +1  # PosHeader(*args_header)
            elif A[i, j] < 0:
                sign_link = -1  # NegHeader(*args_header)
            else:
                raise RuntimeError("Abnormal state has been reached in "
                                   "_compute_graphics_links.")

            color_link = _rgb_to_hex((100, 100, 100, 100))


        # If header exists, it should be removed,
        # because the sign of signal flow can be different
        # from the original sign of header.
        if 'HEADER' in data:
            data.pop('HEADER')

        data['SIGN'] = sign_link
        data['FILL_COLOR'] = color_link

        if f == 0:
            data['WIDTH'] = lw_min
        elif (flow_max - flow_min) == 0:
            data['WIDTH'] = 0.5 * (lw_max + lw_min)
        else:
            log_f = np.log10(np.abs(f))
            log_f = np.clip(log_f, a_min=flow_min, a_max=flow_thr)
            lw = (log_f - flow_min) / (flow_max - flow_min) * (
            lw_max - lw_min) + lw_min
            data['WIDTH'] = lw

