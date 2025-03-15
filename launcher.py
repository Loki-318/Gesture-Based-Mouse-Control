#!/usr/bin/env python3
"""
Gesture Control Application Launcher
This script launches the gesture control application
"""

import sys
import os
from gesture_control_app import main

if __name__ == "__main__":
    # Ensure working directory is set correctly
    if getattr(sys, 'frozen', False):
        # Running as executable
        os.chdir(os.path.dirname(sys.executable))
    
    # Start the application
    main()