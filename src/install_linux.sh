#!/bin/bash

echo "Installing required libraries for mkEqSFC..."
pip install pandas pyyaml numpy matplotlib scipy
sudo apt-get install python3-tk
pip install ttkthemes
echo "Installation complete."
