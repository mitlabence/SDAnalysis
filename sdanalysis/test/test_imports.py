import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)


def test_imports():
    import custom_io as cio
    # import two_photon_session as tps
