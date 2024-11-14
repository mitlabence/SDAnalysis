"""
test_imports.py - Test if imports work properly
"""
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)


def test_imports():
    """Test if the import works properly in the project structure"""
    import custom_io as cio

    # import two_photon_session as tps
    assert True
