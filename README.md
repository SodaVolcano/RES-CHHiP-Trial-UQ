# Uncertainty Quantification for CT Prostate Images

Code for performing uncertainty quantification on CT prostate images.

The repository contains modules for loading DICOM and NIFTI files, preprocessing the images, training a model, and performing uncertainty quantification.

# Setup

## Poetry

Poetry is used to manage and configure Python dependencies. You can run the following command.

```bash
poetry shell   # Enter the poetry venv environment
poetry install  # Install the dependencies

exit # Exit the poetry shell
```

## Devenv

**Warning: CUDA not supported for this option (idk how to fix :'( )**

[Devenv](https://devenv.sh/) is used to create isolated development shells where the dependencies are declared in `devenv.nix` file with input channels defined in `devenv.yaml` and are version-locked in `devenv.lock`. Dependencies and programs installed in the shell are only accessible in the shell. It is internally powered by Nix where the list of Nix packages can be found at [NixOS Packages](https://search.nixos.org/packages).

First, install `devenv`. Then, from the top-most directory of the project, run the following command to let `devenv` install the required dependencies locally, install the requirements, and activate the virtual environment.

```bash
devenv shell
exit # Exit the devbox shell
```


# Usage

## Example Scripts

Example usages of the functions in `uncertainty` is found under `scripts/`.

## Configuration

The project provides preset configurations. You can pass in your own configuration/constants if you do not wish to use the preset values.

| Configuration      | Defined in                  |
| ------------------ | --------------------------- |
| Data, model, and training configurations | `uncertainty.config` |
| Global constants   | `uncertainty.constants`     |

## Logging

Logging is disable by default. To enable logging, add the following lines to your code and logs will be added to `logs/` each time the code is run. You can manually configure the logger if you do not wish to use the provided configurations.

```python
from loguru import logger
from uncertainty.utils.logging import config_logger

logger.enable("uncertainty")
config_logger()
```

## Loading Data

**TIP: As preprocessing may take significant time to complete, consider saving all preprocessed data as h5 files and loading them instead of loading directly from the original files.**

### DICOM

The list of utility functions for reading DICOM files is provided in `uncertainty.data.dicom`, the functions assume that all of the DICOM files for a single patient is stored in a single folder. A usage example is provided below.

**Warning: Preprocessing will take a significant amount of time and RAM usage, do not enable if you don't have a beefy computer (e.g. 32+GB of RAM).**

```python
import uncertainty.data.dicom as dicom

# Load the volume and mask as a PatientScan object
# Method is the interpolation method, see scipy.ndimage.interpn
scan = dicom.load_patient_scan("/path/to/dicom/patient_1/", method="linear", preprocessing=True)
# Note: an iterator is returned, must evaluate by consuming it
scans = dicom.load_patient_scans("/path/to/patients/")
# Load only the mask without interpolation and snake_case name conversion
mask = dicom.load_mask("/path/to/dicom/files", preprocessing=False)

# Saves all DICOM files for all patients as PatientScan objects in parallel
dicom.save_dicom_scans_to_h5("/path/to/patients", "/output/path/", preprocess=True, n_workers=8)
```

### Data Preprocessing

```python
from uncertainty.data.dicom import load_patient_scans
from uncertainty.training.util import preprocess_data, augment_dataset, construct_augmentor

augmentor = construct_augmentor()
pipe(
    load_patient_scans("path/to/scan"),
    preprocess_dataset,
    augment_data(augmentor=augmentor)
)
```

# Tests

From the root directory, run the following command to run the tests:

```bash
pytest
```

---

# Code Overview

This section is only relevant if you wish to use and maintain the codebase.

## Devenv and Nix

Devenv is internally powered by [Nix](https://nixos.org/) which both a package manager and a pure, turing-complete functional programming language. It's used to install packages in isolation from other packages to achieve reproducibility. `devenv.nix` is written in the Nix programming language, you can find the references [here](https://devenv.sh/reference/options/).

## Coding Paradigm

_Majority_ of the data preprocessing code is written in the functional programming paradigm. This means each function is stateless and (ideally) have no side effects [^1]; they only produce an output. Each function consists of a single pipeline of functions where a given input is passed through a series of functions. This is intended to make the logic simpler without the need to keep track of the state and follow non-linear logic paths such as loops and nested conditionals.

A function can by "curried" if it's decorated by `@curry`, where the function can be called with only partial arguments and return a new function that can be called with the remaining arguments. Consider the following example:

```python
from uncertainty.utils.wrappers import curry  # wrapper around toolz.curry

@curry
def add(a, b, c=3):
    return a + b + c

# If some positional arguments are not provided, a function is returned instead
add_5 = add(5)  # equivalent to lambda b: 5 + b + 3
# We can call this function with the remaining argument
add_5(3) # 11
# You can also just use the function normally
add(5, 3)  # 11
# will also return a function because only one positional argument is provided
add(5, c=6)  # equivalent to lambda b: 5 + b + 6
```

[^1]: Technically most functions in `uncertainty` _do_ have side effects in the form of logging (if enabled). Some functions are purely used for their side effects such as read/write operations or displaying a progress bar.
