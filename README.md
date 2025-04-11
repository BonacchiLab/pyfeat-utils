# pyfeat-utils

This repo will implements a CLI interface to a bunch of utils for tunning the py-feat library (pfu for py-feat utils).
The main goal is to make it easier to use the library and to provide a way to process bulk data and perform standardized analyses.

## Installation

You can use uv to create a virtual environment and install the requirements.
py-feat requires Python 3.9.

```bash
uv venv
.\venv\Scripts\activate
uv pip install -r .\requirements.txt
```

To test the installation, run the following command:

```bash
python -c "from feat import Detector"
```

To install with dev tools, run the following command:

```bash
uv venv
.\venv\Scripts\activate
uv pip install -r .\requirements-dev.txt
```

If you see no errors, the installation was successful.
!!! note
    First time the Detector is imported it may take a while to load the model. This is normal.

## Usage
