import time
from contextlib import contextmanager
import datetime
import os 

ROOT_FOLDER = '/scratch/ss05548'
@contextmanager
def measure_import_time(module_name):
    start_time = time.time()
    yield
    end_time = time.time()
    import_time = end_time - start_time
    print(f"Import time for {module_name}: {import_time:.6f} seconds")


def save_errdata_to_file(data_dict, directory="."):
    # Get the current datetime
    current_datetime = datetime.datetime.now()
    
    # Format the datetime as YYMMDD_HHMM
    filename = current_datetime.strftime("%y%m%d_%H%M.json")
    
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    
    # Open the file in append mode and write the data
    with open(file_path, "a") as file:
        file.write(str(data_dict))