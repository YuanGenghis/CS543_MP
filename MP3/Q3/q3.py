import matplotlib.pyplot as plt
import numpy as np
import cv2  
import os

from PIL import Image
import pickle
from sympy import *
from skimage.transform import *
from IPython.display import display

def get_input_lines(im, min_lines=3):
    """
    Allows user to input line segments; computes centers and directions.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        min_lines: minimum number of lines required
    Returns:
        n: number of lines from input
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        centers: np.ndarray of shape (3, n)
            where each column denotes the homogeneous coordinates of the centers
    """
    n = 0
    lines = np.zeros((3, 0))
    centers = np.zeros((3, 0))

    plt.figure()
    plt.axis('off')
    plt.imshow(im)
    print(f'Set at least {min_lines} lines to compute vanishing point')
    print(f'The delete and backspace keys act like right clicking')
    print(f'The enter key acts like middle clicking')
    while True:
        print('Click the two endpoints, use the right button (delete and backspace keys) to undo, and use the middle button to stop input')
        clicked = plt.ginput(2, timeout=0, show_clicks=True)
        if not clicked or len(clicked) < 2:
            if n < min_lines:
                print(f'Need at least {min_lines} lines, you have {n} now')
                continue
            else:
                # Stop getting lines if number of lines is enough
                break

        # Unpack user inputs and save as homogeneous coordinates
        pt1 = np.array([clicked[0][0], clicked[0][1], 1])
        pt2 = np.array([clicked[1][0], clicked[1][1], 1])
        # Get line equation using cross product
        # Line equation: line[0] * x + line[1] * y + line[2] = 0
        line = np.cross(pt1, pt2)
        lines = np.append(lines, line.reshape((3, 1)), axis=1)
        # Get center coordinate of the line segment
        center = (pt1 + pt2) / 2
        centers = np.append(centers, center.reshape((3, 1)), axis=1)

        # Plot line segment
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')

        n += 1

    return n, lines, centers

def plot_lines_and_vp(ax, im, lines, vp):
    """
    Plots user-input lines and the calculated vanishing point.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        vp: np.ndarray of shape (3, )
    """
    bx1 = min(1, vp[0] / vp[2]) - 10
    bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    by1 = min(1, vp[1] / vp[2]) - 10
    by2 = max(im.shape[0], vp[1] / vp[2]) + 10
    
    ax.imshow(im)
    for i in range(lines.shape[1]):
        if lines[0, i] < lines[1, i]:
            pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        else:
            pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        pt1 = pt1 / pt1[2]
        pt2 = pt2 / pt2[2]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')

    ax.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    ax.set_xlim([bx1, bx2])
    ax.set_ylim([by2, by1])

def get_vanishing_point(lines):
    """
    Solves for the vanishing point using the user-input lines.
    """
    u, s, vh = np.linalg.svd(lines.T)
    p = vh[-1]
    return p

def get_horizon_line(pz, px):
    """
    Calculates the ground horizon line.
    """
    line = np.cross(pz, px)
    line = line / np.sqrt(line[0]**2 + line[1]**2)
    return line

def plot_horizon_line(ax, im, horizon_line):
    """
    Plots the horizon line.
    """
    height, width = im.shape[0:2]
    line_x, line_y, line_z = horizon_line
    y1 = - line_z / line_y
    y2 = - (line_z + width * line_x) / line_y
    ax.plot([0, width], [y1, y2], 'r')
    ax.set_ylim([2500, -500])
    ax.imshow(im)

def get_camera_parameters(vpts):
    """
    Computes the camera parameters. Hint: The SymPy package is suitable for this.
    """
    v1, v2, v3 = vpts.T[:,:,np.newaxis]
    f, px, py = 'f', 'px', 'py'
    M = Matrix([[f, 0, px], [0, f, py], [0, 0, 1]])
    A = M.inv().T * M.inv()
    res = nsolve(((v1.T * A * v2), (v2.T * A * v3), (v3.T * A * v1)), 
                 (f, px, py), 
                 (1200, 0, 0))
    focal, u, v = res
    k = M.subs({f:focal, px:u, py:v})
    return focal, u, v, k

def get_homography(im, R):
    """
    Compute homography for transforming the image into fronto-parallel 
    views along the different axes.
    """
    return R @ im

def get_rotation_matrix(K, vpts):
    """
    Computes the rotation matrix using the camera parameters.
    """
    R = np.linalg.inv(K) @ vpts
    norm = np.sqrt(np.sum(R**2, axis=0))
    R = R.T / norm[:, np.newaxis]
    return R.T


im = np.asarray(Image.open('./eceb.jpg'))

# Also loads the vanishing line data if it exists in data.pickle file. 
# data.pickle is written using snippet in the next cell.
if os.path.exists('./data.pickle'):
    with open('./data.pickle', 'rb') as f:
        all_n, all_lines, all_centers = pickle.load(f)
    num_vpts = 3

# ----------------Part1---------------------------------------------------
print('----------------Part1---------------------------------------------------')
# Computing vanishing points for each of the directions
vpts = np.zeros((3, num_vpts))


for i in range(num_vpts):
    fig = plt.figure(); ax = fig.gca()
    
    # <YOUR CODE> Solve for vanishing point
    vpts[:, i] = get_vanishing_point(all_lines[i])
    
    # Plot the lines and the vanishing point
    plot_lines_and_vp(ax, im, all_lines[i], vpts[:, i])
    fig.savefig('Q3_vp{:d}.pdf'.format(i), bbox_inches='tight')


# ----------------Part2---------------------------------------------------
print('----------------Part2---------------------------------------------------')
# <YOUR CODE> Get the ground horizon line
horizon_line = get_horizon_line(vpts[:, 1], vpts[:, 0])
print(horizon_line)

# <YOUR CODE> Plot the ground horizon line
fig = plt.figure(); ax = fig.gca()
plot_horizon_line(ax, im, horizon_line)
fig.savefig('Q3_horizon.pdf', bbox_inches='tight')


#----------------Part3---------------------------------------------------
print('----------------Part3---------------------------------------------------')
# <YOUR CODE> Solve for the camera parameters (f, u, v)
f, u, v, K = get_camera_parameters(vpts)
print(u, v, f)
display(K)
K = np.array(K.tolist()).astype(np.float64)

# ----------------Part4---------------------------------------------------
print('----------------Part4---------------------------------------------------')
# <YOUR CODE> Solve for the rotation matrix
R = get_rotation_matrix(K, vpts)
print(R)