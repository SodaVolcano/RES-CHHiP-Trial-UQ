# Uncertainty Quantification for CT Prostate Images
Code for performing uncertainty quantification on CT prostate images.

The repository contains modules for loading DICOM and NIFTI files, preprocessing the images, training a model, and performing uncertainty quantification.

# Setup
Run the following command to install the required packages:
```bash
make init
```

# Usage
TODO

# Tests
From the root directory, run the following command to run the tests:
```bash
make test
```

# Code Overview
This section is only relevant if you wish to use and maintain the codebase.

## Coding Paradigm
*Majority* of the code is written in pure functional programming paradigm, this means each function is stateless and (ideally) have no side effects [^1]; they only produce an output. Each function only consist of a single pipeline of functions where a given input is passed through a series of functions. This is intended to make the logic simpler without the need to keep track of state and follow non-linear logic paths such as loops and nested conditionals.

A function can by "curried" if it's decorated by `@curry`, where the function can be called with only partial arguments and return a new function that can be called with the remaining arguments. Consider the following example:
```python
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

Some pipelines uses `_`, which is a shortcut for simple lambda functions from `fn.py`. For example:
```python
_      # lambda x: x
_ + 3  # lambda x: x + 3
_.some_method() # lambda x: x.some_method()

# NOTE: below will not work as expected
bool(_)  # bool(lambda x: x), can't pass value to inside functions
_ is not None  # (lambda x: x) is not None, always True as a function is never None
```

[^1]: Technically most functions *do* have side effects in the form of logging (if enabled). Some functions are purely used for their side effects such as read/write operations or displaying a progress bar.