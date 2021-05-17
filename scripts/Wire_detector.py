# Makes legacy code work. Don't judge me
import os
import sys

top_dir = os.path.dirname(os.path.abspath(__file__)) + os.sep
sys.path.append(top_dir + ".." + os.sep)
from Wire_detector import Wire

# class Wire(W):
#     def __init__(self):
#         pass
