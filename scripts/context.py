"""
Path modification to resolve package name

add `from .context import uncertainty` to test modules
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uncertainty
import uncertainty.data
import uncertainty.models
import uncertainty.config
import uncertainty.constants
import uncertainty.training
import uncertainty.utils
