"""
File locking utilities for thread-safe file operations
"""

import fcntl
import os
import time
from contextlib import contextmanager

@contextmanager
def file_lock(file_path, timeout=10):
    """
    Context manager for file locking
    
    Args:
        file_path: Path to the file to lock
        timeout: Maximum time to wait for lock (seconds)
    
    Yields:
        File handle with exclusive lock
    """
    lock_file = f"{file_path}.lock"
    lock_handle = None
    start_time = time.time()
    
    try:
        # Try to acquire lock
        while time.time() - start_time < timeout:
            try:
                lock_handle = open(lock_file, 'w')
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (IOError, OSError):
                time.sleep(0.1)
        else:
            raise TimeoutError(f"Could not acquire lock for {file_path} within {timeout} seconds")
        
        yield lock_handle
        
    finally:
        if lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
                os.remove(lock_file)
            except:
                pass

class ThreadSafeCSVWriter:
    """Thread-safe CSV writer with file locking"""
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def write_row(self, data_dict):
        """Write a row to CSV with file locking"""
        import csv
        
        with file_lock(self.file_path):
            file_exists = os.path.isfile(self.file_path)
            
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data_dict.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data_dict)