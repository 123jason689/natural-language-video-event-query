import sys
import os

# --- PATH HACK START ---
# This adds the current directory to Python's path.
# It ensures that internal absolute imports inside the library 
# (like 'from utils.distributed import ...') continue to work 
# even when you import the library as 'import mobileviclip'.
current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)
# --- PATH HACK END ---

# Expose the submodules so you can do 'from mobileviclip import models'
from . import models
from . import utils