# Install

v0.1.0 targets both Python 2 and Python 3 and depends on `numpy`,
`scipy`, `pandas`, `networkx`, plus the `six` and `future` shims for
the cross-version compatibility layer.

Download the repository:

```bash
$ git clone https://github.com/dwgoon/sfa.git sfa
```

Install from the cloned directory:

```bash
$ cd sfa
$ python setup.py install
```

To pick up updates from the repository without re-installing, use the
`develop` option:

```bash
$ python setup.py develop
```

After that, `git pull origin master` is enough to keep the package
current.

If you do not have permission to the global `site-packages` directory,
use the `--user` flag:

```bash
$ python setup.py install --user
```

Otherwise, consider a
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html).

!!! note "Modern Python / NumPy"
    On NumPy 1.20+, the v0.1.0 code raises `AttributeError` because
    aliases such as `np.float`, `np.int`, and `np.mat` were removed.
    Use the v0.2.x release for Python 3.7+ with modern NumPy/pandas,
    or pin `numpy<1.20` to run v0.1.0 as shipped.
