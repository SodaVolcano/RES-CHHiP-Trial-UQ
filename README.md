# Uncertainty Quantification for CT Prostate Images
Code for performing uncertainty quantification on CT prostate images.

The repository contains modules for loading DICOM and NIFTI files, preprocessing the images, training a model, and performing uncertainty quantification.

# Setup

## Devbox + direnv
This is the recommended method for Linux and Mac OS as the resulting environment is isolated from the rest of the host environment.

**Warning: `direnv` allow any arbitrary bash commands to be executed, please inspect `.envrc` before allowing direnv!**

[Devbox](https://github.com/jetify-com/devbox) is used to create isolated development shells where the dependencies are declared in `devbox.json` file and are version-locked in `devbox.lock`. Dependencies and programs installed in the shell are only accessible in the shell. It is internally powered by Nix where the list of Nix packages can be found at [Nixhub.io](https://www.nixhub.io/). 

[`direnv`](https://github.com/direnv/direnv) is used to extend the current shell by loading and unloading environmental variables automatically as the user enters the current directory. This is used to activate the Devbox shell automatically using the `.envrc` file.

First, install both [Devbox](https://github.com/jetify-com/devbox) and [`direnv`](https://github.com/direnv/direnv). Then, from the top-most directory of the project, run the following command to let Devbox install the required dependencies locally, run pip to install the requirements, and activate the virtual environment.

```bash
direnv allow    # Allow direnv to execute .envrc
```


## pip
Using python version 3.12.4, run the following commands.
```bash
python3 -m venv venv   # Create a virtual environment folder
source venv/bin/activate   # on Mac OS and Linux, OR
venv\Scripts\activate      # On Windows
pip install -r /path/to/requirements.txt   # Install python dependencies
```

You can deactivate the virtual environment by running `deactivate`.


# Usage
## Configuration
Global configuration is defined in `uncertainty.common.constants`. You can pass in your own configuration/constants if you do not wish to use the preset values.

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


# Tests
From the root directory, run the following command to run the tests:
```bash
pytest
```

---

# Code Overview
This section is only relevant if you wish to use and maintain the codebase.

## Devbox and Nix 
Devbox is internally powered by [Nix](https://nixos.org/) which both a package manager and a pure, turing-complete functional programming language. It's used to install packages in isolation from other packages to achieve reproducibility.

[Nix Flakes](https://nixos.org/) is a standarised way of building packages consisting of mainly two attributes. `inputs` specify the dependencies for building the `outputs` where the specified dependencies are version locked in a `flake.lock` file to ensure that they are reproducible on future installs (`devbox.lock` functions in the same manner). The `outputs` attribute is a *function* that takes dependencies in `inputs` to produce an attribute set, such as a set containing `packages` or `devShells`.

Devbox allows packages to be installed from a `flake.nix` file, whether locally or online e.g. via GitHub repositories. The line
```json
"packages": [
    ...
    "github:GuillaumeDesforges/fix-python/"
],
```
simply installs `fix-python` defined in `flake.nix` from the provided GitHub repository.

As Nix installs packages in isolation from one another, many libraries that link to the C/C++ standard libraries such as `glibc` and `zlib` in some assumed paths will not function. `fix-python` is used to fix these paths where, given a list of Nix packages (in `.nix/libs.nix`), it will resolve all references to those packages in the installed python modules at `venv`. This can also be done manually if `fix-python` fails by installing said package in `devbox.json` and exporting the `LD_LIBRARY_PATH` environment variable to `.devbox/nix/profile/default` which contains the installed packages.


## Coding Paradigm
*Majority* of the code is written in the pure functional programming paradigm. This means each function is stateless and (ideally) have no side effects [^1]; they only produce an output. Each function only consist of a single pipeline of functions where a given input is passed through a series of functions. This is intended to make the logic simpler without the need to keep track of state and follow non-linear logic paths such as loops and nested conditionals.

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

[^1]: Technically most functions in `uncertainty` *do* have side effects in the form of logging (if enabled). Some functions are purely used for their side effects such as read/write operations or displaying a progress bar.