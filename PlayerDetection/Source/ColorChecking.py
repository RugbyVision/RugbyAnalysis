import cv2 as cv
import numpy as np
import colorsys as cs

def is_color_in_range(rgb_col, min_hue, max_hue, min_sat, max_sat, min_val, max_val):
    hsv_col = cs.rgb_to_hsv(rgb_col[0] / 255., rgb_col[1] / 255., rgb_col[2] / 255.)

    min_hue = min_hue / 360.
    max_hue = max_hue / 360.

    if min_hue <= hsv_col[0] <= max_hue and \
            min_sat <= hsv_col[1] <= max_sat and \
            min_val <= hsv_col[2] <= max_val:
        return True
    else:
        return False

def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)

def detect_color(img, field_color):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    height, width, _ = np.shape(img)

    data = np.reshape(img, (height * width, 3))
    data = np.float32(data)

    num_clusters = 2
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.)
    flags = cv.KMEANS_PP_CENTERS
    cpc, lbl, centers = cv.kmeans(data, num_clusters, None, criteria, 10, flags)

    bars = []
    rgb_values = []

    for row in centers:
        bar, rgb = create_bar(10, 10, row)
        bars.append(bar)
        rgb_values.append(rgb)

    color = []

    for row in rgb_values:
        if not is_color_in_range(row, *field_color):
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