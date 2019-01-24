#!/usr/bin/env python3
"""Utility functions and classes
"""

import json

class Params():
    """Class to load hyperparameters from a json file.
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Save parameters to json file at json_path
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file at json_path
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
