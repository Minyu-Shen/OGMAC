import torch
import random
import numpy as np
import pandas as pd
import numpy as np

# used for storing figures


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def get_pareto(avg_heads, avg_holds):
    # Initialize the Pareto frontier
    pareto_frontier = []
    # Sort the data based on the first list (ascending order)
    data = sorted(zip(avg_holds, avg_heads))
    # Initialize the maximum value of the second list to track dominated points
    max_y = float('inf')
    # Iterate through the sorted data to find the Pareto front
    for point in data:
        if point[1] <= max_y:
            max_y = point[1]
            pareto_frontier.append(point)
    # Separate the Pareto frontier into separate lists for plotting
    pareto_x = [point[0] for point in pareto_frontier]
    pareto_y = [point[1] for point in pareto_frontier]

    return pareto_x, pareto_y


def get_up_and_downstream_stop_ids(stop_id, stop_num):
    if stop_num % 2 == 0:
        upstr_stop_num = int(stop_num / 2)
        downs_stop_num = int(stop_num / 2) - 1
    else:
        upstr_stop_num = int((stop_num-1) / 2)
        downs_stop_num = int((stop_num-1) / 2)
    upstr_stop_ids = []
    for i in range(1, upstr_stop_num+1):
        if stop_id - i >= 0:
            upstr_stop_ids.append(stop_id - i)
        else:
            upstr_stop_ids.append(stop_num + stop_id - i)
    downs_stop_ids = []
    for i in range(1, downs_stop_num+1):
        if stop_id + i <= stop_num - 1:
            downs_stop_ids.append(stop_id + i)
        else:
            downs_stop_ids.append(stop_id + i - stop_num)
    return upstr_stop_ids, downs_stop_ids


def generate_color(value):
    """
    Generate a color between green (0) and red (1) based on the given value.

    Args:
        value (float): A number between 0 and 1.

    Returns:
        tuple: A tuple representing the RGB color in the format (R, G, B),
               where R, G, and B values are between 0 and 255.
    """
    # Ensure the value is within the valid range [0, 1]
    value = max(0.0, min(1.0, value))

    # Interpolate between green and red based on the value
    red = int(255 * value)
    green = int(255 * (1 - value))

    # Create the RGB color tuple
    color = (red/255, green/255, 0)

    return color


def generate_gray_color(shade):
    """
    Generate a gray color based on the given shade.

    Args:
        shade (float): A number between 0 and 1 representing the shade of gray,
                      with 0 being white and 1 being black.

    Returns:
        tuple: A tuple representing the RGB color in the format (R, G, B),
               where R, G, and B values are between 0 and 255.
    """
    # Ensure the shade is within the valid range [0, 1]
    shade = max(0.0, min(1.0, shade))

    # Calculate the gray intensity by scaling the shade to the range [0, 255]
    gray_intensity = int(255 * (1.0 - shade))

    # Create the RGB color tuple with equal R, G, and B values
    color = (gray_intensity / 255.0, gray_intensity /
             255.0, gray_intensity / 255.0)

    return color
# # Example usage:
# shade = 0.5  # Adjust this value between 0 and 1
# color = generate_gray_color(shade)
# print(f"Shade: {shade}, Color: {color}")
