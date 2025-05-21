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

To use this repository, follow these steps:

1. **Prepare your data**  
   Place all the files you want to process (images, videos, CSVs, TXTs, etc.) inside the `input_data` folder located in the root of the repository.  
   Supported formats include images (`.jpg`, `.jpeg`, `.png`), videos (`.mp4`, `.avi`, `.mov`), and tabular data (`.csv`, `.txt`).

2. **Configure processing options (optional)**  
   If needed, edit the `template_config.json` file to adjust the input folder path or processing options (such as which data types to process).

   **Note:**  
   By default, the provided `template_config.json` is configured for the repository to be located in your `Documents` folder on your PC:
   ```json
   "input_data": "~/Documents/pyfeat-utils/input_data"
   ```
   If you move the repository to another location, update this path accordingly in the JSON file.

3. **Run the processing script**  
   To process your data and generate outputs/statistics, run:

   ```bash
   python -m pyfeat_utils.pyfeat_processor
   ```

   This will process all files in `input_data` according to your configuration and save the results (such as predictions and statistics) in the same folder.

4. **Generate descriptive statistics**  
   This repo also has a script that generates descriptive statistics (mean, median, quartiles, most common emotion, etc.) from the processed data. For this option run:

   ```bash
   python -m pyfeat_utils.descriptive_statistics
   ```

   This will read all processed CSVs in `input_data` and output a summary file called `statistics_summary.csv` in the same folder.

---

**Note:**  
- Make sure your virtual environment is activated before running the scripts.
- The first time you run the scripts, model downloads may take a while.
- All outputs and summaries will be saved in the `input_data` folder for easy access.
