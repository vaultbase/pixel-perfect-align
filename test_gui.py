#!/usr/bin/env python3
"""
Test the GUI without running full alignment
"""

import sys
from PyQt6.QtWidgets import QApplication
from align_gui import PixelPerfectAlignGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Pixel Perfect Align Test")
    
    window = PixelPerfectAlignGUI()
    window.show()
    
    # Test logging
    window.log("GUI Test Mode - Alignment disabled")
    window.status_label.setText("Test Mode - GUI Only")
    
    sys.exit(app.exec())