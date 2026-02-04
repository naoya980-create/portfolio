@echo off
REM Mohs hardness analysis: .py -> .ipynb -> .html for portfolio
cd /d "%~dp0"
jupytext mohs_hardness_analysis.py -o mohs_hardness_analysis.ipynb
if errorlevel 1 exit /b 1
jupyter nbconvert --to html mohs_hardness_analysis.ipynb --output-dir .. --output mohs_hardness.html
if errorlevel 1 exit /b 1
echo Done. Output: ..\mohs_hardness.html
