import time
from contextlib import contextmanager

@contextmanager
def measure_import_time(module_name):
    start_time = time.time()
    yield
    end_time = time.time()
    import_time = end_time - start_time
    print(f"Import time for {module_name}: {import_time:.6f} seconds")
