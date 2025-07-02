#!/bin/bash

echo "Creating virual enviromnet for python..."
python3 -m venv venv

echo "Activating envivorment"
source venv/bin/activate

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing packages"
pip install -r requirements.txt

echo "Virtual envivorment is set up and ready to use"
