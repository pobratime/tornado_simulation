@echo off
echo Creating virtual environment for Python...
python -m venv venv

echo Activating environment...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing packages...
pip install -r requirements.txt

echo Virtual environment is setup and ready to use.
