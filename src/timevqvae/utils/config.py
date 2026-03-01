import yaml


def load_yaml_param_settings(yaml_fname: str):
    """Load hyper-parameter settings from a YAML file."""
    with open(yaml_fname, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
