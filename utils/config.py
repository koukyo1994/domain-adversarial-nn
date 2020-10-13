import yaml


def load_config(path: str):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config
