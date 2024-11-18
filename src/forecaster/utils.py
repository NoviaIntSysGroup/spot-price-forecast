import os
import re
import string


def get_project_root():
    """
    It returns the path to the project root

    :return: The path to the root of the project.
    """
    utils_path = os.path.dirname(os.path.abspath(__file__))
    root_end_idx = list(re.finditer("src", utils_path))[-1].start()
    root_path = utils_path[0:root_end_idx]
    return root_path

def get_safe_filename(filename):
    """
    It returns a safe filename by replacing spaces with underscores and removing special characters.

    Args:
    - filename (str): The filename to be sanitized.

    Returns:
    - str: The sanitized filename.
    """
    return filename.strip().translate(str.maketrans(string.punctuation + " ", '_' * (len(string.punctuation) + 1))).lower()
