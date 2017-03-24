
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


def visualize_signal_flow(net, alg, data, act,
                          color_up=None, color_dn=None,
                          pct_up=50, pct_dn=50):
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

    alg : sfa.Algorithm
        Algorithm object which has the weight matrix, W.
    data : sfa.Data
        Data object which has the n2i and i2n dictionaries.

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

    pct_up : int
        Percentile of up-regulated activity, which is
        the maximum value for color_up.
        Default value is 50.

    pct_dn : int
        Percentile of down-regulated activity, which is
        the maximum value for color_dn. Absolute values of
        down-regulated activities are used.
        Default value is 50.

    Returns
    -------
    None
    """

    W = alg.W
    n2i = data.n2i
    i2n = data.i2n
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

    pos_act = act[act > 0]
    neg_act = np.abs(act[act < 0])

    if pos_act.size == 0:
        pos_cut = 1.0
    else:
        pos_cut = np.percentile(pos_act, pct_up)

    if neg_act.size == 0:
        neg_cut = 1.0
    else:
        neg_cut = np.percentile(neg_act, pct_dn)

    arr_t = np.zeros_like(act)

    for i, elem in enumerate(act):

        if elem > 0:
            t = np.clip(elem / pos_cut, a_min=0, a_max=1)
        elif elem <= 0:
            t = np.clip(np.abs(elem) / neg_cut, a_min=0, a_max=1)

        arr_t[i] = t
        # print(data.i2n[i], elem, neg_cut, np.abs(elem)/neg_cut, t)



    for iden, node in net.nodes.items():
        idx = n2i[iden]

        radius = 20  # + 10*np.log(1+b[idx])
        node.width = node.height = radius

        fold = act[idx]

        if fold > 0:
            color = color_white + arr_t[idx] * (color_up - color_white)
            node['FILL_COLOR'] = QColor(*color)
        elif fold <= 0:
            color = color_white + arr_t[idx] * (color_dn - color_white)
            node['FILL_COLOR'] = QColor(*color)

        node['BORDER_COLOR'] = '#555555'
        _update_single_label_name(net, node, node.name)
        _update_single_label_activity(net, node, fold)
    # end of for : update nodes and labels

    _update_links(net, W, act, i2n)



def _update_links(net, W, act, i2n):
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
        color_link = None
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
            link.width = 1
        else:
            link.width = 0.5*abs(np.log(f))


def _update_single_label_name(net, node, name):
    label = net.labels[name]
    rect = label.boundingRect()
    node.width = 1.1 * rect.width()
    node.height = 1.1 * rect.height()
    label.setPos(-rect.width() / 2, -rect.height() / 2)  # center

    lightness = QColor(node['FILL_COLOR']).lightness()
    label['TEXT_COLOR'] = Qt.black
    label['FONT_SIZE'] = 12
    label['FONT_FAMILY'] = 'Tahoma'
    if lightness < 200:
        label['TEXT_COLOR'] = Qt.white
    else:
        label['TEXT_COLOR'] = Qt.black


def _update_single_label_activity(net, node, x):
    iden = '%s_act' % node.iden.upper()
    str_x = '%f' % (x)
    if iden not in net.labels:
        label_x = TextLabel(node, text=str_x)
        label_x.iden = iden
    else:
        label_x = net.labels[iden]
        label_x.text = str_x % (x)

    label_x.font = QFont('Tahoma', 12)
    label_x['TEXT_COLOR'] = QColor(20, 20, 20)
    rect = label_x.boundingRect()
    rect_ln = net.labels[node.name].boundingRect()
    pos_x = 0.5 * max(rect_ln.width(), node.width)
    label_x.setPos(pos_x, -rect.height() / 2)

    if iden not in net.labels:
        net.add_label(label_x)

