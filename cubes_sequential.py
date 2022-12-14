import numpy as np
from scipy import spatial
from math import pi, cos, sin, asin
from scipy.spatial.transform import Rotation as R



def x_rotation_matrix(angle: float):
    return np.array([
        [1,0,0],
        [0, cos(angle), sin(angle)],
        [0, -sin(angle), cos(angle)]
    ])

def y_rotation_matrix(angle: float):
    return np.array([
        [cos(angle), 0, -sin(angle)],
        [0,1,0],
        [sin(angle), 0, cos(angle)]
    ])

def z_rotation_matrix(angle: float):
    return np.array([
        [cos(angle), sin(angle), 0],
        [-sin(angle), cos(angle), 0],
        [0,0,1]
    ])

def RandomShadow(cube: np.ndarray):
    tmp_cube = np.copy(cube)
    
    x_help_rot = x_rotation_matrix(np.random.uniform(0,2*pi))

    # For mathematically more precise execution use the commented out part instead
    y_help_rot = y_rotation_matrix(np.random.uniform(0,2*pi))
    z_help_rot = z_rotation_matrix(np.random.uniform(0,2*pi))

    rot_matrix = R.from_matrix(x_help_rot @ y_help_rot @ z_help_rot)

    # if you use the more precise part comment everything from here to the comment above

    """
    y_help_rot = y_rotation_matrix(asin(cos(np.random.uniform(0,pi))))
    help_rot = x_help_rot @ y_help_rot

    rot_axis = np.array([1, 0, 0])
    rot_axis = rot_axis @ help_rot

    rot_matrix = R.from_rotvec(np.random.uniform(0,2*pi)*rot_axis)
    """

    # Rotation and projection in one
    tmp_cube = rot_matrix.apply(tmp_cube) [:,:2]

    hull = spatial.ConvexHull(tmp_cube)
    return hull.volume

cube = np.array([[-0.5,-0.5,-0.5],
        [-0.5,-0.5,0.5],
        [-0.5,0.5,-0.5],
        [-0.5,0.5,0.5],
        [0.5,-0.5,-0.5],
        [0.5,-0.5,0.5],
        [0.5,0.5,-0.5],
        [0.5,0.5,0.5]
    ])

print(cube)
shadows = []

for i in range(1000):
    shadows.append(RandomShadow(cube))

avg = np.average(np.array(shadows))
print(avg)