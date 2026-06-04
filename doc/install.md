# Install

Currently, we recommend installing from distributed repositories such as GitHub.
First, download a recent version of the repository as follows.

```bash
$ git clone https://github.com/dwgoon/sfa.git sfa
```

Now, you can install the SFA Python package from the cloned directory.

```bash
$ cd sfa
$ pip install .
```

To also use the `sfa.plot` matplotlib-based helpers, include the `plot`
extra so that `matplotlib` and `seaborn` are installed alongside:

```bash
$ pip install .[plot]
```

If you want to easily update the most recent stable version of the package
from the repository, install in editable mode.

```bash
$ pip install -e .
```

Now, running `git pull origin main` is enough to update the package
from the repository.

If you do not have permission to the global `site-packages` directory,
you can use the `--user` flag:

```bash
$ pip install --user .
```

Otherwise, you can also consider
[Python virtual environments](https://docs.python.org/3/tutorial/venv.html).
