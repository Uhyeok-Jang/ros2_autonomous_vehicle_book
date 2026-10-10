import os
import types
import marshal
from pathlib import Path

def get_path(file_name=None):
    return str(Path(__file__).resolve().parent / file_name)

def get_pyc(module_file):
    file_path = get_path(module_file)
    print('\n파일명:', file_path, '\n\n')
    pyc = open(file_path, 'rb').read()
    code = marshal.loads(pyc[16:])
    module = types.ModuleType('module_name')
    exec(code, module.__dict__)
    return module

camera_perception_func_lib = get_pyc("camera_perception_func_lib.cpython-310.pyc")
