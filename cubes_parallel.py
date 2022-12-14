import numpy as np
import numba as nb

from math import pi,sqrt,cos, sin
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


@cuda.jit
def generate_rotation(rng_states, x_angles, y_angles, z_angles):
    i = cuda.grid(2)[0]

    x_angle = xoroshiro128p_uniform_float32(rng_states, i)*2*pi
    p = cos(x_angle)
    q = sin(x_angle)
    
    # rotational matrix since multidimensional arrays are not fun in cuda
    x_angles[i][0][0] = 1
    x_angles[i][0][1] = 0
    x_angles[i][0][2] = 0
    x_angles[i][1][0] = 0
    x_angles[i][1][1] = p
    x_angles[i][1][2] = q
    x_angles[i][2][0] = 0
    x_angles[i][2][1] = -q
    x_angles[i][2][2] = p

    y_angle = xoroshiro128p_uniform_float32(rng_states, i)*2*pi
    p = cos(y_angle)
    q = sin(y_angle)

    y_angles[i][0][0] = p
    y_angles[i][0][1] = 0
    y_angles[i][0][2] = q
    y_angles[i][1][0] = 0
    y_angles[i][1][1] = 1
    y_angles[i][1][2] = 0
    y_angles[i][2][0] = -q
    y_angles[i][2][1] = 0
    y_angles[i][2][2] = p

    z_angle = xoroshiro128p_uniform_float32(rng_states, i)*2*pi
    p = cos(z_angle)
    q = sin(z_angle)

    z_angles[i][0][0] = p
    z_angles[i][0][1] = q
    z_angles[i][0][2] = 0
    z_angles[i][1][0] = -q
    z_angles[i][1][1] = p
    z_angles[i][1][2] = 0
    z_angles[i][2][0] = 0
    z_angles[i][2][1] = 0
    z_angles[i][2][2] = 1


@cuda.jit
def combine_rotation(mat1, mat2, destination):
    # matrix multiplication
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                destination[cuda.grid(2)[0]][i][j] += mat1[cuda.grid(2)[0]][i][k] * mat2[cuda.grid(2)[0]][k][j]


@cuda.jit (device=True)
def VectorMatrixMultiplication(matrix, vector, result):
    for i in range(3):
        for j in range(3):
            result[i] += matrix[i][j] * vector[j]

@cuda.jit
def rotate_and_project_unit_cube(rotation, src, dest):
    i = cuda.grid(2)[0]

    for j in range(len(src[i])):
        result = cuda.local.array(3, dtype=nb.f4)
        for k in range(3):
            result[k] = 0.0
        VectorMatrixMultiplication(rotation[i], src[i][j], result)

        for k in range(3):
            dest[i][j][k] = result[k]
        
@cuda.jit
def calculate_convex_hull(inp, tmp, result):
    i = cuda.grid(2)[0]

    # sort by distance to 0,0,n
    n = len(inp[i])
    for j in range(n):
        for k in range(0, n - j - 1):
            if (inp[i][k][0]**2 + inp[i][k][1]**2) > (inp[i][k + 1][0]**2 + inp[i][k + 1][1]**2):
                inp[i][k][0], inp[i][k + 1][0] = inp[i][k + 1][0], inp[i][k][0]
                inp[i][k][1], inp[i][k + 1][1] = inp[i][k + 1][1], inp[i][k][1]

    n = len(tmp[i])

    # lose the two points closest to origin since they are overshadowed (pun intended)
    for j in range(n):
        tmp[i][j][0] = inp[i][j + 2][0]
        tmp[i][j][1] = inp[i][j + 2][1]

    # sort by x values
    for j in range(n):
        for k in range(0, n - j - 1):
            if (tmp[i][k][0]) > (tmp[i][k + 1][0]):
                tmp[i][k][0], tmp[i][k + 1][0] = tmp[i][k + 1][0], tmp[i][k][0]
                tmp[i][k][1], tmp[i][k + 1][1] = tmp[i][k + 1][1], tmp[i][k][1]

    # hardcoded convex hull, for reference please consider BILD
    if tmp[i][0][1] < tmp[i][1][1]:
        if tmp[i][1][1] < tmp[i][2][1]:
            for j in range(2):
                result[i][0][j] = tmp[i][0][j]
                result[i][1][j] = tmp[i][1][j]
                result[i][2][j] = tmp[i][2][j]
                result[i][5][j] = tmp[i][3][j]
                result[i][4][j] = tmp[i][4][j]
                result[i][3][j] = tmp[i][5][j]

        else:
            for j in range(2):
                result[i][0][j] = tmp[i][0][j]
                result[i][1][j] = tmp[i][1][j]
                result[i][5][j] = tmp[i][2][j]
                result[i][2][j] = tmp[i][3][j]
                result[i][4][j] = tmp[i][4][j]
                result[i][3][j] = tmp[i][5][j]
    else:
        if tmp[i][1][1] > tmp[i][2][1]:
            for j in range(2):
                result[i][0][j] = tmp[i][0][j]
                result[i][5][j] = tmp[i][1][j]
                result[i][4][j] = tmp[i][2][j]
                result[i][1][j] = tmp[i][3][j]
                result[i][2][j] = tmp[i][4][j]
                result[i][3][j] = tmp[i][5][j]
        else:

            for j in range(2):
                result[i][0][j] = tmp[i][0][j]
                result[i][5][j] = tmp[i][1][j]
                result[i][1][j] = tmp[i][2][j]
                result[i][4][j] = tmp[i][3][j]
                result[i][2][j] = tmp[i][4][j]
                result[i][3][j] = tmp[i][5][j]

@cuda.jit (device=True)
def distance_to_line(point, l1, l2):
    x_diff = l2[0] - l1[0]
    y_diff = l2[1] - l1[1]
    num = abs(y_diff*point[0] - x_diff*point[1] + l2[0]*l1[1] - l2[1]*l1[0])
    den = sqrt(y_diff**2 + x_diff**2)
    return num / den

@cuda.jit (device=True)
def halfway_point(p1, p2, result_point):
    p1_to_p2_0 = p2[0]-p1[0]
    p1_to_p2_1 = p2[1]-p1[1]
    result_point[0] = p1[0] + p1_to_p2_0/2
    result_point[1] = p1[1] + p1_to_p2_1/2

@cuda.jit (device=True)
def distance_between_points(p1, p2):
    return complex(sqrt(abs(p1[0]-p2[0])**2+abs(p1[1]-p2[1])**2)).real

@cuda.jit
def calculate_shadow_area(hexagons, shadow_areas):

    # Calculation of hexagonal shadow
    # for reference please consider BILD
    i = cuda.grid(2)[0]
    points = hexagons[i]
    helper_point = cuda.local.array(2, dtype=nb.f4)
    helper_point[0],helper_point[1] = 0,0
    halfway_point(points[5], points[3], helper_point)

    height_upper = distance_to_line(points[1], points[0], points[2])
    hight_middle = distance_to_line(helper_point, points[0], points[2])
    hight_lower = distance_between_points(helper_point, points[4])
    para_length = distance_between_points(points[0], points[2])

    shadow_areas[i] = para_length*(hight_middle+(height_upper+hight_lower)/2)


g_size = 128
b_size = 16
no_of_items = g_size * b_size

zero_matrix = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]

# unit cube centered arount origin
cube = [
    [-0.5,-0.5,-0.5],
    [-0.5,-0.5,0.5],
    [-0.5,0.5,-0.5],
    [-0.5,0.5,0.5],
    [0.5,-0.5,-0.5],
    [0.5,-0.5,0.5],
    [0.5,0.5,-0.5],
    [0.5,0.5,0.5]
]

zero_cube = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]

hexagon = [
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0]
]

x_rot = np.array([zero_matrix for i in range(no_of_items)])
y_rot = np.array([zero_matrix for i in range(no_of_items)])
z_rot = np.array([zero_matrix for i in range(no_of_items)])
help_rot = np.array([zero_matrix for i in range(no_of_items)])
help2_rot = np.array([zero_matrix for i in range(no_of_items)])

# context for random numbers
rng_states = create_xoroshiro128p_states(g_size * b_size, seed=1)

generate_rotation[g_size, b_size](rng_states, x_rot, y_rot, z_rot)


combine_rotation[g_size, b_size](x_rot, y_rot, help_rot)
combine_rotation[g_size, b_size](help_rot, z_rot, help2_rot)

dest = np.array([zero_cube for i in range(no_of_items)])
rotated = np.array([cube for i in range(no_of_items)])

rotate_and_project_unit_cube[g_size,b_size](help2_rot, rotated, dest)

hull = np.array([hexagon for i in range(no_of_items)])
tmp = np.array([hexagon for i in range(no_of_items)])

calculate_convex_hull[g_size, b_size](dest, tmp, hull)
print(hull)

shadow_areas = np.zeros(no_of_items)

calculate_shadow_area[g_size, b_size](hull, shadow_areas)

print(shadow_areas)

avg = np.average(np.array(shadow_areas))
print(avg)