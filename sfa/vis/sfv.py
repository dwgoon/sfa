
import numpy as np

from qtpy.QtGui import QFont
from qtpy.QtGui import QColor
from qtpy.QtCore import QPointF
from qtpy.QtCore import Qt

from sfv.graphics import LabelClassFactory
from sfv.graphics import HeaderClassFactory

NegHeader = HeaderClassFactory.create('HAMMER')
PosHeader = HeaderClassFactory.create('ARROW')
TextLabel = LabelClassFactory.create('TEXT_LABEL')


def visualize_signal_flow(net, W, n2i, act,
                          color_up=None, color_dn=None,
                          pct=50, dlw=1.0, plw=1.0,
                          show_act=True,
                          fmt_act='%.5f',
                          fix_node_size=False,
                          fix_act_label=False):
    """Visualize signal flow using SFV.

    SFV (Seamless Flow Visualization) is a light-weight,
    programming-oriented python package to visualize
    graphs and networks.

    This function is used in the SFV function,
    'execute(nav, net)', which is called in SFV program.

    Parameters
    ----------
    net : sfv.graphics.Network
        Network object that is given by sfv.

    W : numpy.ndarray
        Weight matrix of the algorithm object.
    n2i : dict
        Name to index dictionary.

    act : numpy.ndarray
        Change in the activities. It is usually calculated
        as x_perturb - x_control, where x is
        the steady-state of activities.

    color_up : numpy.ndarray or QtGui.QColor, optional
        Default is blue (i.e., QColor(0, 0, 255)),
        if it is None.

    color_dn : numpy.ndarray or QtGui.QColor, optional
        Default is red (i.e., QColor(255, 0, 0)),
        if it is None.
    pct : int, optional
        Percentile of activity, which is used to set
        the maximum value for coloring nodes.
        Default value is 50.

    dlw : float, optional
        Link width for unchanged flow, which is 1.

    plw : float, optional
        Parameter for adjusting link width.

    show_act : bool, optional
        Show activity label or not.

    fmt_act : str, optional
        Format string for activity label.

    fix_node_size : bool, optional
        Change the size of node or not.
        Default is False.

    fix_act_label : bool, optional
        Change the graphics of activity label or not.
        The activity value is changed irrespective of
        this parameter. Default is False.

    Returns
    -------
    None
    """

    i2n = {val:key for key, val in n2i.items()}
    color_white = np.array([255, 255, 255])

    if not color_up:
        color_up = np.array([255, 0, 0])
    elif isinstance(color_up, QColor):
        color_up = np.array([color_up.red(),
                             color_up.green(),
                             color_up.blue()])
    else:
        raise ValueError("color_up should be 3-dimensional np.ndarray "
                         "or QtGui.QColor")

    if not color_dn:
        color_dn = np.array([0, 0, 255])
    elif isinstance(color_dn, QColor):
        color_dn = np.array([color_dn.red(),
                             color_dn.green(),
                             color_dn.blue()])
    else:
        raise ValueError("color_dn should be 3-dimensional np.ndarray "
                         "or QtGui.QColor")

    abs_act = np.abs(act)
    thr = np.percentile(abs_act, pct)
    thr = 1 if thr == 0 else thr

    arr_t = np.zeros_like(act)

    for i, elem in enumerate(act):
        t = np.clip(np.abs(elem)/thr, a_min=0, a_max=1)
        arr_t[i] = t



    for iden, node in net.nodes.items():
        idx = n2i[iden]

        if not fix_node_size:
            radius = 20
            node.width = node.height = radius

        fold = act[idx]

        if fold > 0:
            color = color_white + arr_t[idx] * (color_up - color_white)
        elif fold <= 0:
            color = color_white + arr_t[idx] * (color_dn - color_white)

        color = np.int32(color)
        node['FILL_COLOR'] = QColor(*color)

        node['BORDER_COLOR'] = '#555555'
        _update_single_label_name(net, node, node.name,
                                  fix_node_size,)

        if show_act:
            _update_single_label_activity(net, node, fold, fmt_act,
                                          fix_act_label)
        else:
            iden_label = '%s_act' % iden.upper()
            if iden_label in net.labels:
                net.remove_label(net.labels[iden_label])
    # end of for : update nodes and labels

    _update_links(net, W, act, i2n, plw, dlw)



def _update_links(net, W, act, i2n, plw, dlw):
    F = W * act
    i2n = i2n
    ir, ic = W.nonzero()
    for i, j in zip(ir, ic):
        tgt = i2n[i]
        src = i2n[j]
        f = F[i, j]
        link = net.nxdg.edge[src][tgt]['VIS']

        header_old = link.header
        args_header = header_old.width, header_old.height, header_old.offset
        #color_link = None
        if f > 0:
            header = PosHeader(*args_header)
            color_link = QColor(255, 10, 10, 100)
        elif f < 0:
            header = NegHeader(*args_header)
            color_link = QColor(10, 10, 255, 100)
        else:
            if W[i, j]>0:
                header = PosHeader(*args_header)
            elif W[i, j]<0:
                header = NegHeader(*args_header)
            else:
                raise RuntimeError("The logic is abnormal.")

            color_link = QColor(100, 100, 100, 100)

        link.header = header
        link['FILL_COLOR'] = color_link
        f = np.abs(f)
        if f == 0:
            link.width = dlw
        else:
            link.width = max(plw*abs(np.log(f)), dlw)


def _update_single_label_name(net, node, name,
                              fix_node_size):
    label = net.labels[name]

    lightness = QColor(node['FILL_COLOR']).lightness()
    label['TEXT_COLOR'] = Qt.black
    label['FONT_SIZE'] = 10
    label['FONT_FAMILY'] = 'Arial'
    if lightness < 200:
        label['TEXT_COLOR'] = Qt.white
    else:
        label['TEXT_COLOR'] = Qt.black

    rect = label.boundingRect()
    label.setPos(-rect.width() / 2, -rect.height() / 2)  # center
    if not fix_node_size:
        node.width = 1.1 * rect.width()
        node.height = 1.1 * rect.height()


def _update_single_label_activity(net, node, x, fmt, fix_act_label):
    iden = '%s_act' % node.iden.upper()
    str_x = fmt % (x)
    if iden not in net.labels:
        label_x = TextLabel(node, text=str_x)
        label_x.iden = iden
    else:
        label_x = net.labels[iden]
        label_x.text = str_x % (x)

    if not fix_act_label:
        label_x.font = QFont('Arial', 10)
        label_x['TEXT_COLOR'] = QColor(20, 20, 20)
        rect = label_x.boundingRect()
        rect_ln = net.labels[node.name].boundingRect()
        pos_x = node.width/2 + 0.5
        label_x.setPos(pos_x, -rect.height() / 2)

    if iden not in net.labels:
        net.add_label(label_x)



"""<Deprecated documentation>
    pct_up : int, optional
        Percentile of up-regulated activity, which is
        the maximum value for color_up.
        Default value is 50.

    pct_dn : int, optional
        Percentile of down-regulated activity, which is
        the maximum value for color_dn. Absolute values of
        down-regulated activities are used.
        Default value is 50.
"""