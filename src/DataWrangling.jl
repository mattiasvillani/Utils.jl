# Importing the pickle in Python using PyCall.jl #FIXME: This needs to be executed manually. What is going on? 
# I think this is obsolete now, there is a package with pickle?
py"""
import pickle
"""

# Make julia function that pickles (reads a Python file with data)
"""
    unpickle(filename)

Read a Python data file using Python's Pickle via PyCall.jl.
"""
function unpickle(filename)
    return py"pickle.load(open($filename + '.pkl', 'rb'))"
end