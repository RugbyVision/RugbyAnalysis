import cv2 as cv
import numpy as np
import colorsys

def is_color_in_range(rgb_color, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value):
    # Convert RGB to HSV
    hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255., rgb_color[1] / 255., rgb_color[2] / 255.)

    min_hue = min_hue / 360.
    max_hue = max_hue / 360.

    if min_hue <= hsv_color[0] <= max_hue and \
            min_saturation <= hsv_color[1] <= max_saturation and\
            min_value <= hsv_color[2] <= max_value:
        return True
    else:
        return False

def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

def detect_color(img, field_color):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    height, width, dim = np.shape(img)
    # print(height, width)

    data = np.reshape(img, (height * width, 3))
    data = np.float32(data)

    number_clusters = 2
    crriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.)
    flags = cv.KMEANS_PP_CENTERS
    compactness, labels, centers = cv.kmeans(data, number_clusters, None, critera, 10, flags)
    
    bars = []
    rgb_values = []

    for index, row in enumerate(centers):
        bar, rgb = create_bar(10, 10, row)
        bars.append(bar)
        rgb_values.append(rgb)

    color = []

    for index, row in enumerate(rgb_values):
        if not is_color_in_range(row, *filed_color):
            color.append(row)
    return color

def get_ranged_groups(data, groups):
    sorted_data = {}
    for key, crange in groups.items():
        group = []
        for c in data:
            for r in crange:
                if is_color_in_range(c, *r):
                    group.append(c)
        if len(group) > 0:
            sorted_data[key] = group
    return sorted_data