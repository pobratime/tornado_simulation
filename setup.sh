#!/bin/bash

echo "Creating virual enviorment for python..."
python3 -m venv venv

echo "Activating enviorment"
source venv/bin/activate

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing packages"
pip install -r requirements.txt

echo "Virtual enviorment is setup and ready to use"
