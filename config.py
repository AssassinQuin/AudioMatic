import yaml


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


config_path = "config.yaml"
config = load_config(config_path)

# 访问配置项
batch_size_s = config["classify"]["batch_size_s"]
max_segment_duration = config["cut"]["max_segment_duration"]
