import numpy as np
import cv2

### Parameters ###
WIDTH, HEIGHT = 800, 600
N_STATES = 3
V_MAX = 7
EXTENSION_FACTOR = 7

# Change index 0-33 to change starting point of car in 'normal' configuration
INDEX = 0

## Defition of barriers ##

image = cv2.imread('images/track.png')
image = cv2.resize(image, (800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Extract contours with minimum length to prevent other lines except for the contours of the track to be selected
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
min_contour_length = 200
long_contours = [contour for contour in contours if cv2.arcLength(contour, True) > min_contour_length]

# Outer and inner contour of the track
epsilon_outer = 0.001 * cv2.arcLength(long_contours[0], True)
approx_contours_outer = cv2.approxPolyDP(long_contours[0], epsilon_outer, True)
approx_contours_outer = np.squeeze(approx_contours_outer)

epsilon_inner = 0.001 * cv2.arcLength(long_contours[-1], True)
approx_contours_inner = cv2.approxPolyDP(long_contours[-1], epsilon_inner, True)
approx_contours_inner = np.squeeze(approx_contours_inner)

BARRIERS = [approx_contours_outer, approx_contours_inner]


### Definition of checkpoints ###

CHECKPOINTS = np.array([[396, 555, 392, 477],
                         [314, 480, 312, 558],
                         [253, 480, 244, 559],
                         [184, 550, 200, 470],
                         [165, 461, 102, 507],
                         [160, 441, 91, 412],
                         [176, 442, 189, 364],
                         [229, 373, 245, 450],
                         [237, 365, 310, 388],
                         [229, 349, 281, 291],
                         [175, 305, 237, 259],
                         [144, 226, 224, 229],
                         [232, 209, 176, 155],
                         [256, 199, 260, 123],
                         [286, 208, 329, 141],
                         [331, 247, 385, 194],
                         [379, 301, 428, 241],
                         [455, 261, 434, 337],
                         [482, 268, 517, 336],
                         [482, 255, 558, 268],
                         [470, 225, 547, 205],
                         [451, 166, 527, 163],
                         [531, 139, 463, 97],
                         [555, 129, 566, 51],
                         [582, 141, 628, 78],
                         [605, 172, 676, 148],
                         [618, 229, 691, 217],
                         [622, 288, 697, 284],
                         [624, 349, 701, 354],
                         [628, 411, 703, 414],
                         [621, 446, 684, 481],
                         [595, 464, 633, 532],
                         [539, 468, 550, 547],
                         [485, 472, 489, 547]])

# Definition of starting points and angles
x1 = CHECKPOINTS[:, 0]
y1 = CHECKPOINTS[:, 1]
x2 = CHECKPOINTS[:, 2]
y2 = CHECKPOINTS[:, 3]
x_middle = (x1 + x2) / 2
y_middle = (y1 + y2) / 2

starting_points = []
for i, (x_m, y_m) in enumerate(zip(x_middle, y_middle)):
    index = (i + 1) % (len(x_middle))
    x = (x_m + x_middle[index]) / 2
    y = (y_m + y_middle[index]) / 2
    starting_points.append([x, y])

STARTING_POINTS = np.roll(np.array(starting_points).astype(int), 1, axis=0)

angles = []
for i, (x_m, y_m) in enumerate(zip(x_middle, y_middle)):
    index = (i + 1) % (len(x_middle))
    angle = np.arctan2(y_middle[index] - y_m , x_middle[index] - x_m) * 180 / np.pi
    corrected_angle = (270 - angle % 360) if angle % 360 < 270 else (angle % 360)
    angles.append(corrected_angle)

STARTING_ANGLES = np.roll(np.array(angles).astype(int), 1)


## Definition of different configurations
def get_config(index):
    init_pos = STARTING_POINTS[index]
    init_angle = STARTING_ANGLES[index]
    if index != 0:
        checkpoints = np.vstack((CHECKPOINTS[index:], CHECKPOINTS[:index]))
    elif index == 0:
        checkpoints = CHECKPOINTS
    return init_pos, init_angle, checkpoints

# Straight 1
straight1_init_pos, straight1_init_angle, straight1_checkpoints = get_config(index=32)

STRAIGHT1_CONFIG = {
    'init_pos': straight1_init_pos,
    'init_angle': straight1_init_angle,
    'checkpoints': straight1_checkpoints,
}

# Straight 2
straight2_init_pos, straight2_init_angle, straight2_checkpoints = get_config(index=25)

STRAIGHT2_CONFIG = {
    'init_pos': straight2_init_pos,
    'init_angle': straight2_init_angle,
    'checkpoints': straight2_checkpoints,
}

#Right-curve 1
right1_init_pos, right1_init_angle, right1_checkpoints = get_config(index=20)

RIGHT1_CONFIG = {
    'init_pos': right1_init_pos,
    'init_angle': right1_init_angle,
    'checkpoints': right1_checkpoints,
}

#Right-curve 2
right2_init_pos, right2_init_angle, right2_checkpoints = get_config(index=28)

RIGHT2_CONFIG = {
    'init_pos': right2_init_pos,
    'init_angle': right2_init_angle,
    'checkpoints': right2_checkpoints,
}

#Left-curve 1
left1_init_pos, left1_init_angle, left1_checkpoints = get_config(index=15)

LEFT1_CONFIG = {
    'init_pos': left1_init_pos,
    'init_angle': left1_init_angle,
    'checkpoints': left1_checkpoints,
}

#Left-curve 2
left2_init_pos, left2_init_angle, left2_checkpoints = get_config(index=6)

LEFT2_CONFIG = {
    'init_pos': left2_init_pos,
    'init_angle': left2_init_angle,
    'checkpoints': left2_checkpoints,
}

# Normal configuration
init_pos, init_angle, checkpoints = get_config(INDEX)

NORMAL_CONFIGURATION = {
    'init_pos': init_pos,
    'init_angle': init_angle,
    'checkpoints': checkpoints,
}

CONFIGURATIONS = {
    'straight1': STRAIGHT1_CONFIG,
    'straight2': STRAIGHT2_CONFIG,
    'right1': RIGHT1_CONFIG,
    'right2': RIGHT2_CONFIG,
    'left1': LEFT1_CONFIG,
    'left2': LEFT2_CONFIG,
    'normal': NORMAL_CONFIGURATION
}