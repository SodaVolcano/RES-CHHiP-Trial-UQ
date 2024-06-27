"""
Path modification to resolve package name

add `from .context import uncertainty_quantification` to test modules
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uncertainty
