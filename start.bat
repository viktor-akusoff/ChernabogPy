@echo off

:: Set the name of the virtual environment
set venv_name=.venv

:: Check if Python is installed using the "where" command
where python > nul 2>&1
if %errorlevel% == 0 (
    echo Python is already installed.
) else (
    echo Python is not installed. Attempting to install with winget...
    winget install python3
    if %errorlevel% == 0 (
        echo Python installed successfully.
    ) else (
        echo Failed to install Python with winget. Please install it manually.
        pause
        exit /b 1
    )
)

:: Create the virtual environment if it doesn't exist
if not exist "%~dp0\%venv_name%\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv "%~dp0\%venv_name%"
)

:: Activate the virtual environment
call "%~dp0\%venv_name%\Scripts\activate"

:: Install requirements if needed
echo Installing base packages
pip install tqdm numpy==1.26.4 matplotlib opencv-python


echo Installing PyTorch packages...
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Run main.py
echo Running main.py...
python "%~dp0main.py"

:: Pause the console window to see the output
pause