# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:39:18 2017

@author: K.Ataman
"""

"""
TODO:
	have a special version for importing from online repos
"""
'''
path = "C:/Users/K.Ataman/Documents/GitHub/python_big_data/dataset_.py"
file = "dataset_"
'''
def class_importer(path, file):
    """
    Imports the file named "file" from a specific location "path"
    
    Inputs:
        path = Location of the file. Has the form C:/path/to/file.py
        file = Name of file. No need to put .py at the end
    Outputs:
        contents of the file. For example, if you do 
        foo = class_importer(path, file), all the contents of the file
        can be accessed with foo.*
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(file, path)
    module_name = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_name)
    return module_name
    '''
    you can test if the impor is successful by doing
    foo = class_importer(path, file)
    foo.name_of_class_from_file()
    '''
    