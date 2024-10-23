
import os
import sys
try:
    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))  # SDAnalysis/SDAnalysis folder
    root_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../../'))  # SDAnalysis folder (top-level folder)
    sys.path.insert(0, root_dir)
    sys.path.insert(0, project_dir)

except:
    raise Exception("Exception while adding root_dir to sys.path")
finally:
    from data_documentation import DataDocumentation as DD


def test_data_documentation():
    env_dict = dict()
    fpath_env = os.path.join(root_dir, ".env")
    assert os.path.exists(fpath_env)
    if not os.path.exists(fpath_env):
        print(".env does not exist")
    else:
        with open(fpath_env, "r") as f:
            for line in f.readlines():
                l = line.rstrip().split("=")
                env_dict[l[0]] = l[1]
    assert len(env_dict) > 0
    assert "DATA_DOCU_FOLDER" in env_dict
