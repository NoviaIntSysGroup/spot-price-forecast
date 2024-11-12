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

def save_figure(fig, filename, save_dir):
    """
    Saves a figure to a file.

    Args:
    - fig (matplotlib.figure.Figure): The figure to save.
    - filename (str): The name of the file to save the figure to.
    - save_dir (str): The directory to save the figure to (default is None).
    """
    filename = get_safe_filename(filename)
    if save_dir is not None:
        save_path = os.path.join(save_dir, filename)
    else:
        save_path = filename
    fig.savefig(save_path, bbox_inches='tight')


############################
