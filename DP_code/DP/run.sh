#!/bin/bash
# Helper script to run Python scripts with correct PYTHONPATH and venv

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to include the DP directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Use the new Python 3.11 virtual environment
PYTHON="$SCRIPT_DIR/../venv_new/bin/python3"

# Run the python script with all arguments passed to this script
"$PYTHON" "$@"

