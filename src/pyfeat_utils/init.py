#!/usr/bin/env python
# @File: src\init.py
# @Author: Niccolo' Bonacchi (@nbonacchi)
# @Date: Thursday, April 10th 2025, 12:12:50 pm
"""Creates input and output folders in the user's Documents directory under a folder named "pyfeat-utils".
Uses the template config.json file to create a new config.json file in the root folder.
"""

from pathlib import Path
import json

# Get the user's Documents directory
documents_dir = Path.home() / "Documents"
# Create the pyfeat-utils directory in Documents
pyfeat_utils_dir = documents_dir / "pyfeat-utils"
pyfeat_utils_dir.mkdir(parents=True, exist_ok=True)

# Load the template_config.json file from the current directory
template_config_path = Path(__file__).parent / "template_config.json"
with open(template_config_path, "r") as template_config_file:
    template_config = json.load(template_config_file)

# Create the input and output folders from the loaded template_config.json file
input_folder = pyfeat_utils_dir / template_config["data_pyfeat-utils"]
input_folder.mkdir(parents=True, exist_ok=True)


# Copy the template_config.json file to the root folder define absolute paths insted of user home
