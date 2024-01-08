import functools
import os
import time
from contextlib import contextmanager


@contextmanager
def ignoring(*exceptions):
    try:
        yield
    except exceptions:
        pass

PATH = "test.txt"
@contextmanager
def TemporaryPath(path):
    global PATH # explicitly declare global variable
    raw_path = path
    try:
        file_name, file_extension = os.path.splitext(path)
        PATH = file_name + "_temp" + file_extension
        yield PATH
    finally:
        PATH = raw_path
    
@contextmanager
def timing(msg):
    start = time.time()
    yield
    end = time.time()
    print(f"{msg}: {end-start}")
    
@functools.lru_cache(maxsize=None) # memoization
def possible(idea):
    print(f"Hmm {idea}...")
    time.sleep(3)
    return "Very Interesting!"

def main():
    msg = "Running Time"
    idea = "NewJeansNet"
    
    with timing(msg):
        start_time = time.time()
        for i in range(4):
            print(possible(idea))
    
    arr = [1, 2]
    with ignoring(IndexError, ZeroDivisionError):
        print(arr[2])
        # print(array[2])
        
    with TemporaryPath("test.txt"):
        with open(PATH, "w") as f:
            f.write("Hello, @contextmanager!")
    print(PATH)
    
if __name__ == "__main__":
    main()