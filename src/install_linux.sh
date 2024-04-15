#!/bin/bash

echo "Installing required libraries for mkEqSFC..."
pip install pandas pyyaml numpy
sudo apt-get install python3-tk
sudo apt-get install gnuplot
sudo apt-get install gnuplot-x11
pip install ttkthemes
echo "Installation complete."
