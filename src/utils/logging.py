"""
Logging utilities
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logging configuration
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress verbose libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('rawpy').setLevel(logging.WARNING)
    logging.getLogger('imageio').setLevel(logging.WARNING)
    
    return root_logger