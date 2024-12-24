# Uncertainty Quantification for CT Prostate Images
Code for performing uncertainty quantification on CT prostate cancer images.

## Content
- [Setup](#setup)
    - [uv](#uv)
    - [Nix](#nix)
        - [Auto-activation with Direnv (Optional)](#auto-activation-with-direnv-optional)
- [Usage](#usage)
    - [Configuring the Project](#configuring-the-project)
    - [Using `uncertainty` as a Library Module](#using-uncertainty-as-a-library-module)
        - [Logging](#logging)
    - [Using the Scripts](#using-the-scripts)
    - [Tests](#tests)
- [Code Overview](#code-overview)
    - [File Overview](#file-overview)
    - [Coding Paradigm](#coding-paradigm)
        - [Pipe](#pipe)
        - [Curry](#curry)
    - [Auto-wiring of Configuration Dictionary](#auto-wiring-of-configuration-dictionary)


## Setup

You can **either** install directly via [uv](https://docs.astral.sh/uv/) or use [Nix](https://nixos.org/) to indirectly manage uv for extra reproducibility which also offers some convenient aliases.

Instructions below assumes you are at the **top-level of the project directory** (i.e. folder containing `pyproject.toml` etc).

### uv
[uv](https://docs.astral.sh/uv/) manages and configures Python dependencies. First install it following the [installation guide](https://docs.astral.sh/uv/getting-started/installation/). Then, any Python commands can be run by appending `uv run` in front of your command which will automatically download project dependencies, e.g.

```bash
uv run python ./scripts/prepare_dataset.py   # equivalent to running `python ./scripts/prepare_dataset.py`
```

### Nix

**Warning: CUDA not supported for this option, idk how to fix :'(**

[Nix](https://nixos.org/) is a purely functional programming language *and* package manager used to create isolated and reproducible development shells. A `flake.nix` file defines project dependencies and environment which activates the shell defined in `shell.nix`. First, install Nix following the [installation guide](https://nixos.org/download/). Then, start a development shell by running...

```bash
# Enable experimental features `nix-command` and `flakes`, then run the `develop` command
# Or if you've already enabled these features, just run `nix develop`
nix --extra-experimental-features nix-command --extra-experimental-features flakes develop
# you can now run `uv run python ...` etc
```

The Nix shell comes with short-hand alises in `shell.nix` such as `pytest` for `uv run pytest .` etc.


#### Auto-activation with Direnv (Optional)

**Warning: `direnv` allow the execution of any arbitrary bash code in `.envrc`, please examine `.envrc` before you proceed!**

[`direnv`](https://direnv.net/) is used to automatically activate the Nix flake when you enter into the folder containing this repository. First, install it via the [official installation guide](https://direnv.net/docs/installation.html) and [hook it into your shell](https://direnv.net/docs/hook.html) (HINT: run `echo $SHELL` to see what shell you are using). Then, inside the project directory where `.envrc` is in the same folder, run...

```bash
direnv allow  # allow execution of .envrc automatically
direnv disallow # stop automatically executing .envrc upon entering the project folder
```


## Usage

You can either...
1. Run some script from `scripts/` which reads from `configuration.yaml`. If option flags are specified, those values will override values defined in `configuration.yaml`
2. Import `uncertainty` as a module and use in your own code

### Configuring the Project
The project is configured globally via `configuration.yaml` and values in it can be automatically passed to function parameters if they are decorated with `@auto_match_config`. Preset augmentations and preprocessing are defined in `uncertainty`, but you can pass in your own function/transformations to functions/classes that uses them as well.


| Configuration      | Defined in                  |
| ------------------ | --------------------------- |
| Project-wide configuration | `configuration.yaml` |
| Global constants and list of ROI names to include/exclude   | `uncertainty/constants.py`     |
| Augmentations | In `uncertainty/data/augmentations.py`, `augmentations()` and `batch_augmentations()` |
| Data preprocessing | Functions in `uncertainty/data/processing.py` |
| Global constants | Variables in `uncertainty/constants.py` |

### Using `uncertainty` as a Library Module
Just import `uncertainty` lol.

#### Logging
Logging is disable by default. To enable logging, add the following lines to your code.

```python
from loguru import logger

from uncertainty.config import configuration
from uncertainty.utils import config_logger

logger.enable("uncertainty")
config_logger(**configuration())
```

### Using the Scripts

| Script (in order of usage)               | Description                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| `prepare_dataset.py` | Preprocess folder of folders of DICOM files and save in a single HDF5 file.                                     |
| `train_model.py`     | Initialise a training directory, perform data splitting, and train a list of models across $n$ validation folds |
| `evaluate_model.py`      | Evaluate models on the (test) dataset using specified metrics and output `csv` files of the result             |


### Tests
To run tests in `pytest/`, run

```bash
uv run pytest .
```

---

## Code Overview

This section is only relevant if you wish to use and maintain the codebase.

### File Overview

| File(s)/Folder                                 | Description                                                                                                                                                                                                                                                                                                           |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`, `uv.lock`, `.python-version` | First installation option: file containing project list of dependencies and configurations; a version lock file for the dependencies; and a file with Python version used by the project. **If uv is installed**, running `uv run ...` will automatically sync uv to these files (or, run `uv sync` to sync manually) |
| `flake.nix`, `shell.nix`, `flake.lock`         | Second installation option: Nix files for initialising development shell with project dependencies installed (including uv) and fixing Python dynamic library paths. **If Nix is installed**, start the shell with `nix --extra-experimental-features nix-command --extra-experimental-features flakes develop`       |
| `.envrc`                                       | File used by Direnv to automatically start a Nix development shell defined in `flake.nix` upon `cd`-ing into the project directory. **If Direnv is installed**, [hook it into your shell](https://direnv.net/docs/hook.html) and then run `direnv allow`                                                              |
| `devshell.nix`                                 | Nix development shell that installs VS Code with useful extensions (not updated in a while). Activate with `export NIXPKGS_ALLOW_UNFREE=1 && nix-shell ./devshell.nix`                                                                                                                                                                                                                          |
| `run-train.slurm`                              | SLURM script for sending model training jobs to Kaya - high performance computing system at UWA (run `sbatch run-train.slurm`)                                                                                                                                                                                                                       |
| `configuration.yaml`                           | Configuration file of the project containing settings for the data, training, and model hyperparameters                                                                                                                                                                                                               |
| `uncertainty/`                                 | Folder containing project code, can be imported in Python code as a library module                                                                                                                                                                                                                                                                      |
| `tests/`                                       | Tests for `uncertainty/`, run with `uv run pytest ./tests`                                                                                                                                                                                                                                                            |
| `scripts/`                                     | Set of Python scripts using functions from `uncertainty/`, e.g. for preparing the dataset, training model(s), evaluating trained model(s)                                                                                                                                                                             |

### Coding Paradigm

Many parts of the code is written roughly in the **functional programming** paradigm. 

#### Pipe

Given a value `x`, `toolz.pipe` just passes `x` through a series of functions (it's just a for-loop...).

```python
import toolz as tz
from toolz import curried

# Below is equivalent to
# str(curried.get(5)(tz.identity([3, 4] * 4)))

do_nothing = True
tz.pipe(
    [3, 4],   # value
    lambda lst: lst * 4,
    tz.identity if do_nothing else tz.concat,   # tz.identity will be called here
    curried.get(5),  # get(5) is still a FUNCTION, see "curry" below
    str,
)   # OUTPUT: '4'
```

#### Curry

Yummy.

A function can by "curried" if it's decorated with `@curry`. A "curried" function can be called with *only some of the required arguments* (i.e. partially initialised). This is a **new function** that can be called with the remaining arguments.

```python
from uncertainty.utils import curry  # wrapper around toolz.curry

@curry
def add(a, b, c=3):
    return a + b + c

# If some positional arguments are not provided, a function is returned instead
# of raising an error
add_5 = add(5)  # equivalent to lambda b: 5 + b + 3
# We can call this function with the remaining argument
add_5(3) # 11
# You can also just use the function normally
add(5, 3)  # 11
# will also return a function because only one positional argument is provided
add(5, c=6)  # equivalent to lambda b: 5 + b + 6
```

### Auto-wiring of Configuration Dictionary
When `configuration.yaml` is parsed into a Python dictionary, the keys are transformed into the format `<prefix>__<param_name>` where `<prefix>__` is used to distinguish configuration arguments from normal keyword arguments.

A function decorated with `uncertainty.utils.auto_match_config` can accept unpacked configuration dictionary even if the dictionary contained extra keyword arguments not needed by the function.

```python
from uncertainty import auto_match_config

config = {
    "test__a": 69,  # hehe
    "test2__a": .5,
    "test__b": "hello",
    "test__x": 1,
}

# 1. Only dictionary entries with specified prefixes are passed to the function
@auto_match_config(prefixes=["test"])
def test(a, b):
    return a, b

# config["test2__a"] have wrong prefix so is not used
test(**config)  # OUTPUT: (69, "hello")


# 2. Manually specified kwargs override config entries
test(b="no hello >:(", **config)  # OUTPUT: (69, "no hello >:(")
# You must explcitly overwrite config args using keyword params!
test("no hello >:(", **config) # ERROR: duplicate value for param `b`



# 3. If param with same name appear in dictionary, later entries override earlier ones
@auto_match_config(prefixes=["test", "test2"])
def test2(a):
    return a

# (`config["test2__a"]` overrides `config["test__a"]`)
test2(**config)  # OUTPUT: 0.5


# 4. If function have `**kwargs`, the entire `config` is passed to the function
@auto_match_config(prefixes=["test2"])
def test3(a, **kwargs):
    # Passing `config` along to inner functions via `kwargs`
    test(**kwargs)  # OUTPUT: (69, "hello")
    return a

test3(**config)  # OUTPUT: 0.5
```