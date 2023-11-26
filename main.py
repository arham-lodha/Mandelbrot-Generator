import os

import numpy as np
import yaml
import argparse
import matplotlib as plt

from runner_functions import create_image

DEFAULT_IMAGE_CONFIG = {
    "accelerated": True,
    "height": 512,
    "width": 512,
    "center": np.array([-0.75, 0.0]),
    "world_width": 3.5,
    "max_iterations": 100,
    "escape_radius": 2.0,
    "color_scheme": 0,
    "colormap": np.array([[]], dtype=np.uint8),
    "smooth": False,
    "raster": False,
    "mixed_raster": False,
    "fast_quadtree": True,
    "show_quadtree": False,
}

IMAGE_CONFIG_TYPES = [
    ("accelerated", bool),
    ("height", int),
    ("width", int),
    ("center_real", float),
    ("center_imaginary", float),
    ("world_width", float),
    ("max_iterations", int),
    ("escape_radius", float),
    ("color_scheme", int),
    ("builtin_colormap", bool),
    ("colormap", str),
    ("smooth", bool),
    ("raster", bool),
    ("mixed_raster", bool),
    ("fast_quadtree", bool),
    ("show_quadtree", bool)
]


def denormalize(arr):
    """
    Denormalizes a 2D array by scaling each value in the range [0, 1] to the range [0, 255].

    Parameters:
    arr (list[list[float]]): 2D array to denormalize.

    Returns:
    list[list[int]]: Denormalized 2D array.
    """
    return [[int(val * 255) for val in row] for row in arr]


def parse_arguments():
    """
    Parses command line arguments using argparse for the Mandelbrot Set Generator.

    Returns:
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Mandelbrot Set Generator")

    parser.add_argument('directory', nargs='?', default=None,
                        help="Specify the output directory. If not provided, a default directory will be created.")

    parser.add_argument("--video", "-v", action="store_true", help="Create zoom video")

    return parser.parse_args()


def create_default_run_directory():
    """
    Creates a default run directory for the Mandelbrot Set Generator.

    Returns:
    str: The path of the created run directory.
    """
    run_number = 0
    while True:
        run_directory = f'Mandelbrot_run{run_number}'
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)
            return run_directory
        run_number += 1


def get_line_values(line):
    """
    Parses a line of values and returns them as a list of integers.

    Parameters:
    line (str): The input line containing values.

    Returns:
    list[int]: List of integer values parsed from the input line.

    Raises:
    Exception: If the line has more than 3 values or if any value is not an integer or not in the range [0, 255].
    """
    values = line.split()

    if len(values) > 3:
        raise Exception("Line has more than 3 values")

    try:
        # Convert each value to an integer
        ints = [int(value) for value in values]

        # Check if each integer is between 0 and 255
        for num in ints:
            if not (0 <= num <= 255):
                raise Exception("Value in line is not in between 0 and 255")

        return ints

    except ValueError:
        raise Exception("Value in line is not an integer")


def get_colormap(colormap_file):
    """
    Reads a colormap file and returns the colormap as a 2D array of integers.

    Parameters:
    colormap_file (str): Path to the colormap file.

    Returns:
    list[list[int]]: Colormap represented as a 2D array of integers.
    """
    with open(colormap_file, 'r') as file:
        colormap = [get_line_values(line) for line in file]

    return colormap


def set_photo_config(directory):
    """
    Sets the configuration for generating Mandelbrot Set images based on user preferences.

    Parameters:
    directory (str): Output directory for saving images.

    Returns:
    dict: Configuration dictionary for generating Mandelbrot Set images.

    Raises:
    Exception: If the configuration file is not found or has incorrect types, or if there are colormap-related errors.
    """
    config = DEFAULT_IMAGE_CONFIG

    if directory:
        config_file_path = os.path.join(directory, 'config.yml')

        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as config_file:
                user_config = yaml.load(config_file, Loader=yaml.FullLoader)

            for key, val_type in IMAGE_CONFIG_TYPES:
                if key in user_config and not isinstance(user_config[key], val_type):
                    raise Exception(f"Configuration file has incorrect type in {key}")

                if key in user_config:
                    if key == "center_real":
                        config['center'][0] = user_config[key]
                    elif key == "center_imaginary":
                        config['center'][1] = user_config[key]
                    else:
                        config[key] = user_config[key]

            if config['color_scheme'] not in [0, 1, 2, 3]:
                raise Exception("Incorrect color scheme set. Color scheme must be 0, 1, 2, or 3")

            if config['color_scheme'] == 3:
                if 'builtin_colormap' not in config:
                    raise Exception(f"If colormap colorscheme is chosen, "
                                    f"you must indicate if built in color map is being used")

                if config['builtin_colormap']:
                    colormaps = plt.colormaps

                    if config['colormap'] not in colormaps:
                        raise Exception(f"Provided colormap is not built in.")

                    config['colormap'] = np.array(denormalize(colormaps[config['colormap']].colors), dtype=np.uint8)

                else:
                    colormap_file_path = os.path.join(directory, config['colormap'])

                    if not os.path.exists(colormap_file_path):
                        raise Exception(f"Error: Colormap file doesn't exist")

                    config['colormap'] = np.array(get_colormap(colormap_file_path), dtype=np.uint8)

    return config


def main():
    """
    Main function for the Mandelbrot Set Generator. Parses command line arguments, creates a default run directory
    if necessary, and generates Mandelbrot Set images based on the specified configuration.
    """
    args = parse_arguments()

    directory = args.directory

    if not args.directory:
        directory = create_default_run_directory()

    create_image(set_photo_config(args.directory), directory)


if __name__ == "__main__":
    main()
