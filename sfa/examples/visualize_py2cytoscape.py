# -*- coding: utf-8 -*-

import pickle

import numpy as np
import pandas as pd
import networkx as nx

from py2cytoscape.data.cynetwork import CyNetwork
from py2cytoscape.data.cyrest_client import CyRestClient

import sfa


def calc_width(F, dg, data):
        ir, ic = F.nonzero()
        for i in range(ic.size):
            isrc, itgt = ic[i], ir[i]
            src = data.i2n[isrc]
            tgt = data.i2n[itgt]
            dg.edge[src][tgt]['weight'] = np.abs(F[itgt, isrc])
            dg.edge[src][tgt]['sign'] = np.sign(F[itgt, isrc])
    # end of for

if __name__ == "__main__": 
    
    cond = 'CNT'
    
    algs = sfa.AlgorithmSet()
    alg = algs.create('SP')
    
    ds = sfa.DataSet()
    #ds.create("BORISOV_2009")
    #data = ds["BORISOV_2009"]["15m_AUC_EGF=0.1+I=100"]
    data = ds.create("MOLINELLI_2013")
    
    alg.data = data
    alg.initialize()
    
    N = data.dg.number_of_nodes()
    b = np.zeros((N,), dtype=np.float)
    targets = ['aHDAC']
    inds = []
    vals = []
    
    alg.apply_inputs(inds, vals)
    b[inds] = vals    
    x_cnt = alg.compute(b)
    SF_cnt = alg.W*x_cnt
    
    alg.apply_perturbations(targets, inds, vals)
    b[inds] = vals
    x_ptb = alg.compute(b)
    SF_ptb = alg.W*x_ptb

    cond = cond.lower()
    dg = nx.DiGraph(data.dg)
    if cond == 'cnt':        
        SF = SF_cnt
    else:
        SF = SF_ptb
        
    
    calc_width(SF, dg, data)
    
    #with open("nx_position_object.pydat", "rb") as fin:
    #    dg_pos = pickle.load(fin)
    
    
    # Start cyREST session
    cy = CyRestClient()
    cy.session.delete()
    net = cy.network.create_from_networkx(dg)
   
    # Set layout option
    cy.layout.apply(name='kamada-kawai',
                    network=net)

    # Get CyNetworkView objects from CyNetwork
    view_id_list = net.get_views() 
    view = net.get_view(view_id_list[0], format='view')

    # View as DataFrame makes it easier to see
    # the names of visual properties.
    df_node_view = view.get_node_views_as_dataframe()
    df_edge_view = view.get_edge_views_as_dataframe() 
    
    print(df_node_view.columns)
    print(df_edge_view.columns)
  
    
    for id_node, row in df_node_view.iterrows():
        name = row.NODE_LABEL
        row.NODE_BORDER_PAINT = '#000000'
        row.NODE_BORDER_WIDTH = 2.0
        #row.NODE_FILL_COLOR = '#003DA8'
        row.NODE_LABEL_COLOR = '#FFFFFF'
        row.NODE_LABEL_FONT_SIZE = 12
        row.NODE_X_LOCATION = dg_pos.node[name]['x']
        row.NODE_Y_LOCATION = dg_pos.node[name]['y']
        row.NODE_WIDTH = dg_pos.node[name]['width']
    # end of for
    
    # Update edge data using batch_update
    view.batch_update_node_views(df_node_view)
        
    # It is convenient to use the edge table from CyNetwork object
    # for dealing with edge data such as signs of edges.
    # (The edge table is a Pandas.DataFrame object.)
    table_edges = net.get_edge_table() 
    
    
    # Select options for the shapes of target arrows
    # according to the sign of edge.
    target_arrow_shapes = {}
    edge_weights = {}
    for id_edge, row in table_edges.iterrows():
        id_edge = int(id_edge)
        if row.sign>0:
            target_arrow_shapes[id_edge] = "Delta"
        elif row.sign<0:
            target_arrow_shapes[id_edge] = "T"
        elif row.sign==0:
            pass # Do nothing for unexisting edges
        else:
            err_msg = "The value of sign for edge should be 0, 1, or -1."
            raise ValueError(err_msg)
        # end of if-else
        edge_weights[id_edge] = row.weight
    # end of for

    # Update target arrow shapes of the signed network]
#    view.update_edge_views(visual_property="EDGE_WIDTH",
#                           values=edge_weights)
                           
    view.update_edge_views(visual_property="EDGE_TARGET_ARROW_SHAPE",
                           values=target_arrow_shapes)
    

    
    # Change the widths and colors of edge stroke
    weights = np.log(table_edges.weight)
    weights_tr = weights #weights[np.abs(weights-weights.mean())<=(3*weights.std())]
    min_w = weights_tr.min()
    max_w = weights_tr.max()
    
    max_R = 255
    min_R = 10
    
    val_G = 10
    val_B = 100
    val_R = 200
    
    max_width = 10
    min_width = 1
    
    df_edge_view = view.get_edge_views_as_dataframe()
    for id_edge, row in df_edge_view.iterrows():
        #row.EDGE_WIDTH = 4 #np.random.randint(1, 5)
        weight = edge_weights[id_edge]
#        val_R = int((weight-min_w)*(max_R-min_R) \
#                /(max_w-min_w)+min_R)
        
        xcolor = "#%02x%02x%02x" % (val_R,
                                    val_G,
                                    val_B)
                                    
        row.EDGE_STROKE_UNSELECTED_PAINT = xcolor
        row.EDGE_TARGET_ARROW_UNSELECTED_PAINT = xcolor
        
#        if weight<=min_w:
#            weight = min_w
#        elif weight>=max_w:
#            weight = max_w
            
        val_width = ((weight-min_w)*(max_width-min_width) \
                    /(max_w-min_w)+min_width)
                    
        row.EDGE_WIDTH = val_width
        row.EDGE_TRANSPARENCY = 180
    # end of for
    
    # Update edge data using batch_update
    view.batch_update_edge_views(df_edge_view)
    
    view.update_network_view(visual_property="NETWORK_SCALE_FACTOR",
                             value=1.0)
    

    #from IPython.display import Image

#    fname_fig = "img_%s.png"%(fname_out)
#    fout = open(fname_fig, "wb")    
#    fout.write(net.get_png())
#    fout.close()
