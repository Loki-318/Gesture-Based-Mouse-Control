@echo off
echo Installing requirements...
pip install -r requirements.txt
pip install pyinstaller

echo Building application...
pyinstaller --name="GestureControl" ^
            --windowed ^
            --icon=hand_icon.ico ^
            --add-data="hand_icon.ico;." ^
            --hidden-import=pystray._win32 ^
            --hidden-import=PIL ^
            --hidden-import=mediapipe ^
            --hidden-import=cv2 ^
            launcher.py

echo Build complete. Executable is in the dist/GestureControl folder.
pause