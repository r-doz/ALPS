# commons/config.py
import json
import os

# Get the path to the config.json file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# Load the config data
with open(config_path, 'r') as config_file:
    config = json.load(config_file)