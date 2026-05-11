cd "C:\Users\uitleen 2\Downloads\TOMMY2"

call venv\Scripts\activate.bat

start python run.py

ping 127.0.0.1 -n 11 >nul

start "" http://localhost:5000