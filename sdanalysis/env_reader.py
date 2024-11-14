import os

# TODO: depending on where this function is called from, the path might be different?


def read_env() -> dict:
    # get directory of current file (env_reader.py)
    current_file_path = os.path.abspath(__file__)
    current_file_path = os.path.split(current_file_path)[0]
    # .env is in the parent folder
    env_file_path = os.path.dirname(current_file_path)
    env_file_path = os.path.join(env_file_path, ".env")

    env_dict = dict()
    if not os.path.exists(env_file_path):
        print(".env does not exist")
    else:
        with open(env_file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                l = line.rstrip().split("=")
                env_dict[l[0]] = l[1]
    return env_dict
