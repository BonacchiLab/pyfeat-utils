[![Python Versions](https://img.shields.io/badge/python-3.9-blue)]()
[![Based on py-feat](https://img.shields.io/badge/Based%20on-py--feat-006d44)](https://py-feat.org/pages/intro.html)



# pyfeat-utils

This repo will implements a CLI interface to a bunch of utils for tunning the py-feat library (pfu for py-feat utils).
The main goal is to make it easier to use the library and to provide a way to process bulk data and perform standardized analyses.

For more info about the library, visit https://py-feat.org/pages/intro.html 

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


**Note:**  
    The first time the Detector is imported it may take a while to load the model. This is normal.

## Usage

To use this repository, follow these steps:

1. **Prepare your data**  
   Place all the files you want to process (images, videos, etc.) inside the `data_pyfeat-utils` folder that will be automatically created in your documents.  
   Supported formats include images (`.jpg`, `.jpeg`, `.png`), videos (`.mp4`, `.avi`, `.mov`).

2. **Configure processing options (optional)**  
   If needed, edit the `template_config.json` file to adjust the input folder path or processing options (such as which data types to process).


3. **Run the processing script**  
   To process your data run:

   ```bash
   python -m pyfeat_utils.pyfeat_processor
   ```

   This will process all files in `data_pyfeat-utils` according to your configuration and save the results (such as predictions and statistics) in the same folder.

4. **Generate descriptive statistics**  
   This repo also has a script that generates descriptive statistics (mean, median, quartiles, most common emotion, etc.) from the processed data. For this option run:

   ```bash
   python -m pyfeat_utils.descriptive_statistics
   ```

   This will read all processed CSVs in `data_pyfeat-utils` and output a summary file called `statistics_summary.csv` in the same folder.

---

**Note:**  
- Make sure your virtual environment is activated before running the scripts.
- The first time you run the scripts, model downloads may take a while.
- All outputs and summaries will be saved in the `data_pyfeat-utils` folder for easy access.
