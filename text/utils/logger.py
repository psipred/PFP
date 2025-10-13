"""Logging utilities for capturing terminal output."""

import sys
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm as original_tqdm


class TqdmToLogger(original_tqdm):
    """
    TQDM progress bar that writes to logger instead of stdout.
    This prevents progress bars from cluttering the log file.
    """
    def __init__(self, *args, **kwargs):
        # Force tqdm to only output to console, not to be captured by logger
        kwargs['file'] = sys.stdout
        kwargs['ncols'] = 100
        super().__init__(*args, **kwargs)


class DualOutput:
    """Write to both console and log file."""
    
    def __init__(self, log_file, console_stream):
        self.log_file = log_file
        self.console = console_stream
        self.buffer = ""
    
    def write(self, message):
        # Write to console
        self.console.write(message)
        self.console.flush()
        
        # Write to log file (filter out ANSI escape codes for progress bars)
        if message and not self._is_progress_bar(message):
            self.log_file.write(message)
            self.log_file.flush()
    
    def _is_progress_bar(self, message):
        """Detect if message is from a progress bar."""
        # Check for common progress bar indicators
        indicators = ['\r', '\x1b[', '|', '█', '▏', '▎', '▍', '▌', '▋', '▊', '▉']
        return any(ind in message for ind in indicators) and len(message) < 200
    
    def flush(self):
        self.console.flush()
        self.log_file.flush()
    
    def isatty(self):
        return self.console.isatty()


def setup_logging(log_dir, experiment_name=None):
    """
    Setup dual logging to both console and file.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Optional name for the log file
    
    Returns:
        log_file_path: Path to the created log file
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_filename = f"{experiment_name}_{timestamp}.log"
    else:
        log_filename = f"training_{timestamp}.log"
    
    log_file_path = log_dir / log_filename
    
    # Open log file
    log_file = open(log_file_path, 'w', buffering=1)
    
    # Redirect stdout and stderr to dual output
    sys.stdout = DualOutput(log_file, sys.__stdout__)
    sys.stderr = DualOutput(log_file, sys.__stderr__)
    
    # Write header
    print("=" * 70)
    print(f"Logging started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file_path}")
    print("=" * 70)
    print()
    
    return log_file_path


def restore_stdout():
    """Restore original stdout and stderr."""
    if hasattr(sys.stdout, 'log_file'):
        sys.stdout.log_file.close()
    if hasattr(sys.stderr, 'log_file'):
        sys.stderr.log_file.close()
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class ExperimentLogger:
    """Context manager for experiment logging."""
    
    def __init__(self, log_dir, experiment_name=None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file_path = None
    
    def __enter__(self):
        self.log_file_path = setup_logging(self.log_dir, self.experiment_name)
        return self.log_file_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print("\n" + "=" * 70)
            print("ERROR OCCURRED:")
            print(f"Type: {exc_type.__name__}")
            print(f"Message: {exc_val}")
            print("=" * 70)
        
        print("\n" + "=" * 70)
        print(f"Logging ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        restore_stdout()
        return False  # Don't suppress exceptions