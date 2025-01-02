"""
Path modification to resolve package name

add `from .context import chhip_uq` to test modules
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chhip_uq

# Import aliases
MaskDict = chhip_uq.data.MaskDict
PatientScan = chhip_uq.data.PatientScan
constants = chhip_uq.constants

utils = chhip_uq.utils
training = chhip_uq.training
models = chhip_uq.models
evaluation = chhip_uq.evaluation
data = chhip_uq.data
config = chhip_uq.config
metrics = chhip_uq.metrics

# Patch paths
PATCH_OS_WALK = "os.walk"
PATCH_LIST_FILES = "chhip_uq.utils.path.list_files"
PATCH_DCMREAD = "pydicom.dcmread"
PATCH_RT_CREATE_FROM = "rt_utils.RTStructBuilder.create_from"
PATCH_LISTDIR = "os.listdir"
PATCH_NIBABEL_LOAD = "nibabel.load"

# Some functions use memoize so use different path to avoid caching
path_id = 0


def gen_path():
    global path_id
    path_id += 1
    return f"path/to/folder{path_id}"
