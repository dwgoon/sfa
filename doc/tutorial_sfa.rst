..  -*- coding: utf-8 -*-

Signal flow analysis
====================

This brief tutorial will guide you to start utilizing SFA.
We wrote this tutorial assuming users already have the overall knowledge about
`the original journal paper
 <https://www.nature.com/articles/s41598-018-23643-5>`_.



Creating algorithm object
--------------------------

``sfa.AlgorithmSet`` deals with creating and managing the algorithm objects in SFA.
Thus, we first need to create ``sfa.AlgorithmSet`` object.

.. code-block:: python

    >>> import sfa
    >>> algs = sfa.AlgorithmSet()


Now, we can create algorithm objects with ``sfa.AlgorithmSet``.
Create `Signal Propagation (SP)` algorithm,
which is designated by its abbreviation, ``SP``.

.. code-block:: python

    >>> alg = algs.create('SP')
    SP algorithm has been created.
    >>> alg.abbr
    'SP'
    >>> alg
    SignalPropagation object

As ``sfa.AlgorithmSet`` has the functionality of dictionary,
we can also access the created algorithms using the abbreviations as keys.

.. code-block:: python

    >>> algs['SP']
    SignalPropagation object


Setting hyperparameter values
-----------------------------

Algorithms in SFA have hyperparameters that adjust and constrain
the behavior of the algorithms.
``ParameterSet``, a nested object defined in ``sfa.Algorithm``,
have member variables that contain the information
about the various hyperparameters.
The below shows the examples of the parameters.

.. code-block:: python

    >>> alg.params
    <sfa.algorithms.sp.SignalPropagation.ParameterSet at 0x25b5a7e5550>
    >>> alg.params.alpha
    0.5

We can see the default value of ``alpha`` is 0.5.
``alpha`` is a hyperparameter that controls the proportion of signal flow
in determining the next system state, *x(t+1)*, in the following formula.

.. math::

    x(t+1) = \alpha Wx(t) + (1-\alpha)b

Thus, ``0.5`` means the algorithm reflects the effects of signal flow
on half of estimating *x(t+1)*.

We can easily change the value of ``alpha``
by assigning a real value between 0 and 1.

.. code-block:: python

    >>> alg.params.alpha = 0.5
    0.5
    >>> alg.params.alpha = 0.9
    >>> alg.params.alpha
    0.9


Another hyperparameter is ``apply_weight_norm``,
which designates whether to use link weight normalization.
The default value is ``False``, but it is recommended to set it as ``True``.

.. code-block:: python

    >>> alg.params.apply_weight_norm
    False
    >>> alg.params.apply_weight_norm = True

Refer to the documentation for more details about the other hyperparameters.


Creating data object
--------------------

Creating and handling data objects in SFA are similar to those of algorithms.
A data object is also designated by its abbreviation, as in the algorithm.
For example, the datasets for `Borisov et al. <http://msb.embopress.org/content/5/1/256>`_
can be created using ``BORISOV_2009`` as follows.

.. code-block:: python

    >>> ds = sfa.DataSet()
    >>> mdata = ds.create('BORISOV_2009')
    BORISOV_2009 data has been created.
    >>> mdata  # Multiple datasets.
    {'120m_AUC_EGF=0.001+I=0.1': BorisovData object,
     '120m_AUC_EGF=0.001+I=1': BorisovData object,
     '120m_AUC_EGF=0.001+I=10': BorisovData object,
    ...


The above ``mdata`` or ``ds['BORISOV_2009']`` is a ``dict`` that contains
multiple dataset objects with different conditions.
For example, ``120m_AUC_EGF=0.001+I=0.1`` denotes the dataset was created by
performing a simulation under the stimulation of 0.001M EGF and 0.1M insulin
using the original ODE model, where the activity of a biomolecule was
calculated by estimating the area under the curve (AUC) of the time profile.

We can select a dataset object by using the abbreviation.

.. code-block:: python

    >>> data = mdata['120m_AUC_EGF=0.001+I=0.1']
    >>> data.abbr
    '120m_AUC_EGF=0.001+I=0.1'

We can also consider a utility function in SFA, ``sfa.get_avalue``,
which arbitrarily selects a dataset object from the dictionary.

.. code-block:: python

    >>> data = sfa.get_avalue(mdata)
    >>> data.abbr
    '120m_AUC_EGF=0.001+I=0.1'

Actually, ``sfa.get_avalue`` returns the first item by applying the
`next() <https://docs.python.org/3/library/functions.html#next>`_
built-in fuction to a given ``dict`` object.


Accessing the members of data object
------------------------------------

The data object (instantiated with a subclass of ``sfa.Data``) has
various data structures that are required for using ``sfa.Algorithm``.
For example, ``sfa.Data`` object has the information about network topology in
``A`` (adjacency matrix in ``numpy``'s ndarray_),
``dg`` (``NetworkX``'s DiGraph_),
and ``n2i`` ( ``dict`` for mapping names to the indices of ``A``).


.. code-block:: python

    >>> data.n2i  # Name to index mapper.
    {'AKT': 0,
     'EGF': 1,
     'EGFR': 2,
     'ERK': 3,
     'GAB1': 4,
     'GAB1_SHP2': 5,
     'GAB1_pSHP2': 6,
     'GS': 7,
     'I': 8,
     'IR': 9,
     'IRS': 10,
     'IRS_SHP2': 11,
     'MEK': 12,
     'PDK1': 13,
     'PI3K': 14,
     'PIP3': 15,
     'RAF': 16,
     'RAS': 17,
     'RasGAP': 18,
     'SFK': 19,
     'SHC': 20,
     'mTOR': 21}
    >>> data.A[n2i['ERK'], n2i['MEK']]  # MEK -> ERK
    1
    >>> data.A[n2i['GAB1'], n2i['ERK']]  # ERK -| GAB1
    -1
    >>> data.A[n2i['ERK'], n2i['EGFR']]  # No link between EGFR and ERK.
    0
    >>> for src, trg, attr in data.dg.edges(data=True):
    ...     if attr['SIGN'] > 0:
    ...         print('%s -> %s'%(src, trg))
    ...     elif attr['SIGN'] < 0:
    ...         print('%s -| %s'%(src, trg))
    ...
    AKT -> mTOR
    AKT -| RAF
    EGF -> EGFR
    EGFR -> RasGAP
    EGFR -> SFK
    EGFR -> PI3K
    EGFR -> GAB1
    EGFR -> GAB1_pSHP2
    EGFR -> SHC
    EGFR -> GS
    ERK -| GAB1
    ERK -| GS
    GAB1 -> GAB1_SHP2
    GAB1 -> GAB1_pSHP2
    GAB1 -> PI3K
    GAB1 -> GS
    GAB1 -> RasGAP
    GAB1_SHP2 -> GAB1_pSHP2
    GAB1_SHP2 -| RasGAP
    GAB1_pSHP2 -> GS
    GAB1_pSHP2 -| RasGAP
    GS -> RAS
    I -> IR
    IR -> RasGAP
    IR -> IRS
    IR -> SFK
    IR -> PI3K
    IRS -> IRS_SHP2
    IRS -> GS
    IRS -> PI3K
    IRS_SHP2 -| RasGAP
    MEK -> ERK
    PDK1 -> AKT
    PI3K -> PIP3
    PIP3 -> PDK1
    PIP3 -> IRS
    PIP3 -> GAB1
    RAF -> MEK
    RAS -> RAF
    RasGAP -| RAS
    SFK -> IRS
    SFK -> GAB1
    SFK -> GAB1_pSHP2
    SFK -> RAF
    SHC -> GS
    mTOR -> AKT
    mTOR -| IRS


Analyzing data with algorithm
-----------------------------

To make ``sfa.Algorithm`` work with ``sfa.Data``,
we should first assign the data object to the algorithm object.

.. code-block:: python

    >>> alg.params.alpha = 0.5
    >>> alg.params.apply_weight_norm = True
    >>> alg.data = data  # Assign the data object to the algorithm.
    >>> alg.initilize()  # Initialize the algorithm object.


In the initization of the algorithm (calling ``sfa.Algorithm.initialize``),
the algorithm prepares estimaing signal flow
by performing some necessary tasks such as link weight normalization.

.. code-block:: python

    >>> data.A[data.n2i['GAB1'], data.n2i['EGFR']]
    1
    >>> alg.W[data.n2i['GAB1'], data.n2i['EGFR']]
    0.1889822365046136

Note that the element of the weight matrix is different
from that of adjacency matrix.

One of the important tasks is to determine
the values of the basal activity before analyzing signal flow.
The effects of input stimulation or perturbation are basically reflected to
the basal activity vector, *b*.
For example, EGF stimulation can be reflected to *b* as follows.

.. code-block:: python

    >>> import numpy as np
    >>> N = data.dg.number_of_nodes()  # The number of nodes; data.A.shape[0]
    >>> b = np.zeros((N,), dtype=np.float)
    >>> b[data.n2i['EGF']] = 1

Now, we can perform the estimation of signal flow,
and examine how the two outputs, ERK and AKT, have changed.

.. code-block:: python

    >>> xs1 = alg.compute(b) # xs: x at steady-state
    >>> xs1
    array([0.00155625, 0.5       , 0.25      , 0.00165546, 0.02951243,
           0.00659918, 0.03226491, 0.0367612 , 0.        , 0.        ,
           0.00608503, 0.0017566 , 0.00331091, 0.00401268, 0.02780067,
           0.01390033, 0.00662182, 0.00733528, 0.01601391, 0.03340766,
           0.04724556, 0.00055022])
    >>> xs1[data.n2i['ERK']]
    0.0016554557287082902
    >>> xs1[data.n2i['AKT']]
    0.0015562514037656679

We can see the signs of the two outputs are positive,
which means ERK and AKT are upregulated by EGF stimulation.

Next, let's apply an inhibitory perturbation to the network.
For example, we can perturb MEK by setting its basal activity as follows.

.. code-block:: python

    >>> b[data.n2i['MEK']] = -1
    >>> b[data.n2i['EGF']], b[data.n2i['MEK']]
    (1.0, -1.0)
    >>> b
    array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

    >>> xs2 = alg.compute(b)
    >>> xs2[data.n2i['MEK']]
    -0.4947084519007513
    >>> xs2[data.n2i['ERK']]
    -0.24735422595037565
    >>> xs2[data.n2i['AKT']]
    0.001836795161913794

At this time, the sign of ERK is negative,
which means it is downregulated by MEK inhibition.
On the other hand, AKT is not downregulated
by the inhibition under EGF stimulation.

If we want to examine how the inhibition of MEK affects each node,
we take the difference between the vectors of two results.

.. code-block:: python

    >>> dxs = xs2 - xs1  # Difference between the two results.
    >>> ind_up = np.where(dxs > 0)[0]  # Indices of upregulated nodes
    >>> ind_dn = np.where(dxs < 0)[0]  # Indices of downregulated nodes
    >>> for idx in ind_up:
    ...     print(data.i2n[idx])  # data.i2n: Index to name mapper.
    AKT
    GAB1
    GAB1_SHP2
    GAB1_pSHP2
    GS
    IRS
    IRS_SHP2
    PDK1
    PI3K
    PIP3
    RAF
    RAS
    RasGAP
    mTOR
    >>> for idx in ind_dn:
    ...     print(data.i2n[idx])
    ERK
    MEK


This result shows that only MEK and ERK are upregulated
by the inhibition of MEK under EGF stimulation.


Applying perturbation to link
-----------------------------

In some cases, a perturbation should be reflected to link weight,
not basal activity. For example, if we want to examine what happens
when PI3K cannot send signal to its downstreams
(i.e., the out-links of PI3K are removed).

.. code-block:: python

    >>> b = np.zeros((N,), dtype=np.float)
    >>> b[data.n2i['EGF']] = 1
    >>> alg.W[:, data.n2i['PI3K']] *= 0  # Remove all the out-link weights.
    >>> xs3 = alg.compute(b)
    >>> xs3[data.n2i['ERK']]
    0.00172210494367554
    >>> xs3[data.n2i['AKT']]
    0.0
    >>> dxs = xs3 - xs1  # xs1 is the same as the previously computed one.
    >>> dxs[data.n2i['ERK']]
    6.664921496724974e-05
    >>> dxs[data.n2i['AKT']]
    -0.0015562514037656679

We can see that AKT is downregulated if all out-links of PI3K are lost.


Estimating signal flows
-----------------------
The estimation of signal flow is defined as
the multiplication of link weight and activity of the source node.
The activity is usually is the steady-state activity

.. math::

    F(t)_{ij} = W_{ij} \cdot x(t)_{j}


Follwing the definition, we can compute the signal flow as follows.

.. code-block:: python

    >>> alg.initialize()  # Obtain the intact weight matrix.
    >>> W1 = alg.W.copy()  # Get a copy of the weight matrix.
    >>> F1 = W1*xs1  # Element-wise multiplication of each row and xs1

Note that the above code snippet is not matix-vector multiplication,
but it is element-wise multiplication of vectors (ndarray_ in NumPy_).
The following shows some of the estimated signal flows.

.. code-block:: python

    >>> F1[data.n2i['PIP3'], data.n2i['PI3K']]
    0.02780066830505488
    >>> F1[data.n2i['ERK'], data.n2i['MEK']]
    0.0033109114574165805
    >>> F1[data.n2i['GAB1'], data.n2i['ERK']]
    -0.0005852919858618747


If we want to compare the two conditions,
we can compute the net signal flow as follows.


.. math::

    F_{net} = F_{c2} - F_{c1}


Let's use the PI3K example of "`Applying perturbation to link`_" again.


.. code-block:: python

    >> W3 = alg.W.copy()  # alg.W is the intact one.
    >> W3[:, data.n2i['PI3K']] *= 0  # Apply the PI3K
    >>> F3 = W3*xs3
    >>> Fnet = F3 - F1  # Net signal flow.
    >>> Fnet[:, data.n2i['PI3K']]
    >>> ir, ic = data.A.nonzero()
    >>> for i in range(ir.size):
    ...     idx_trg, idx_src = ir[i], ic[i]
    ...     src = data.i2n[idx_src]
    ...     trg = data.i2n[idx_trg]
    ...     sf = Fnet[idx_trg, idx_src]  # Signal flow
    ...     print("Net signal flow from %s to %s: %f"%(src, trg, sf))
    Net signal flow from PDK1 to AKT: -0.002837
    Net signal flow from mTOR to AKT: -0.000275
    Net signal flow from EGF to EGFR: 0.000000
    Net signal flow from MEK to ERK: 0.000133
    Net signal flow from EGFR to GAB1: 0.000000
    Net signal flow from ERK to GAB1: -0.000024
    Net signal flow from PIP3 to GAB1: -0.004013
    Net signal flow from SFK to GAB1: 0.000000
    Net signal flow from GAB1 to GAB1_SHP2: -0.000903
    Net signal flow from EGFR to GAB1_pSHP2: 0.000000
    Net signal flow from GAB1 to GAB1_pSHP2: -0.000451
    Net signal flow from GAB1_SHP2 to GAB1_pSHP2: -0.000160
    Net signal flow from SFK to GAB1_pSHP2: 0.000000
    Net signal flow from EGFR to GS: 0.000000
    Net signal flow from ERK to GS: -0.000019
    Net signal flow from GAB1 to GS: -0.000368
    Net signal flow from GAB1_pSHP2 to GS: -0.000088
    Net signal flow from IRS to GS: -0.000450
    Net signal flow from SHC to GS: 0.000000
    Net signal flow from I to IR: 0.000000
    Net signal flow from IR to IRS: 0.000000
    Net signal flow from PIP3 to IRS: -0.004013
    Net signal flow from SFK to IRS: 0.000000
    Net signal flow from mTOR to IRS: 0.000195
    Net signal flow from IRS to IRS_SHP2: -0.001102
    Net signal flow from RAF to MEK: 0.000267
    Net signal flow from PIP3 to PDK1: -0.008025
    Net signal flow from EGFR to PI3K: 0.000000
    Net signal flow from GAB1 to PI3K: -0.000451
    Net signal flow from IR to PI3K: 0.000000
    Net signal flow from IRS to PI3K: -0.000551
    Net signal flow from PI3K to PIP3: -0.027801
    Net signal flow from AKT to RAF: 0.000635
    Net signal flow from RAS to RAF: -0.000102
    Net signal flow from SFK to RAF: 0.000000
    Net signal flow from GS to RAS: -0.000327
    Net signal flow from RasGAP to RAS: -0.000027
    Net signal flow from EGFR to RasGAP: 0.000000
    Net signal flow from GAB1 to RasGAP: -0.000368
    Net signal flow from GAB1_SHP2 to RasGAP: 0.000130
    Net signal flow from GAB1_pSHP2 to RasGAP: 0.000088
    Net signal flow from IR to RasGAP: 0.000000
    Net signal flow from IRS_SHP2 to RasGAP: 0.000225
    Net signal flow from EGFR to SFK: 0.000000
    Net signal flow from IR to SFK: 0.000000
    Net signal flow from EGFR to SHC: 0.000000
    Net signal flow from AKT to mTOR: -0.001100


We can see some links have no change
in their signal flows between the two conditions.
Obviously, the signal flow from PI3K to PIP3 has decreased
due to the perturbation.
However, the depletion of all out-links of PI3K has upregulated
the signal flow from MEK to ERK (i.e., positive value).


Creating a dataset with network structure
-----------------------------------------


- Describe how to define own datasets only with network topology.
- Explanation for the members of Data class.



.. _NumPy: http://www.numpy.org
.. _ndarray: https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.ndarray.html
.. _DiGraph: https://networkx.github.io/documentation/networkx-1.10/reference/classes.digraph.html
