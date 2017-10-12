Installing SFA
===============

Currently, we recommend installing from distributed repositories such as GitHub.
First, download a recent version of the repository as follows.

.. code-block:: bash

    $ git clone https://github.com/dwgoon/sfa.git sfa

Now, you can install SFA Python package from the cloned directory.

.. code-block:: bash

    $ cd sfa
    $ python setup.py install

If you want to easily update the most recent stable version of the package
from the repository, use ``develop`` option insead of ``install``.

::

    $ python setup.py develop


Now, running ``git pull origin master`` is enough to update the package
from the repository.

If you don't have permission to the global ``site-packages`` directory,
you can use the following flags: ``--user``,  ``--home``, and ``--prefix``.
Refer to the `official document <https://docs.python.org/3.6/install/index.html>`_
in more details for using the flags.

For example, you can simply install the package with ``--user`` flag.

::

    $ python setup.py install --user


Otherwise, you can also consier
`Python virtual environemnts <https://docs.python.org/3/tutorial/venv.html>`_.
