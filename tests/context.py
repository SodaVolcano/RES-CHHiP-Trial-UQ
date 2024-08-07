"""
Path modification to resolve package name

add `from .context import uncertainty` to test modules
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uncertainty

# Import aliases
Mask = uncertainty.data.mask.Mask
PatientScan = uncertainty.data.patient_scan.PatientScan
c = uncertainty.constants

# utils
list_files = uncertainty.utils.path.list_files
generate_full_paths = uncertainty.utils.path.generate_full_paths
capture_placeholders = uncertainty.utils.string.capture_placeholders
placeholder_matches = uncertainty.utils.string.placeholder_matches
resolve_path_placeholders = uncertainty.utils.path.resolve_path_placeholders

# Patch paths
PATCH_OS_WALK = "os.walk"
PATCH_LIST_FILES = "uncertainty.utils.path.list_files"
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
