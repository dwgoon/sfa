# -*- coding: utf-8 -*-
"""Ensure ``tests/`` is on sys.path so test files can import siblings
like ``_skip_helpers`` without making the directory a package.
"""
import os
import sys

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
