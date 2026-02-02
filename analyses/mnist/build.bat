@echo off
REM MNIST analysis: .py -> .ipynb -> .html for portfolio
cd /d "%~dp0"
jupytext mnist_analysis.py -o mnist_analysis.ipynb
if errorlevel 1 exit /b 1
jupyter nbconvert --to html mnist_analysis.ipynb --output-dir .. --output mnist.html
if errorlevel 1 exit /b 1
echo Done. Output: ..\mnist.html
