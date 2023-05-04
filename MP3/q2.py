from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
import skimage
import skimage.io
import cv2
import scipy
from scipy.spatial import distance

def get_residual(F, p1, p2):
    """
    Function to compute the residual average residual on frame 2
    param: F (3x3): fundamental matrix: (pt in frame 2).T * F * (pt in frame 1) = 0
    param: p1 (Nx2): 2d points on frame 1
    param: p2 (Nx2): 2d points on frame 2
    """
    P1 = np.c_[p1, np.ones((p1.shape[0],1))].transpose()
    P2 = np.c_[p2, np.ones((p2.shape[0],1))].transpose()
    L2 = np.matmul(F, P1).transpose()
    L2_norm = np.sqrt(L2[:,0]**2 + L2[:,1]**2)
    L2 = L2 / L2_norm[:,np.newaxis]
    pt_line_dist = np.multiply(L2, P2.T).sum(axis = 1)
    return np.mean(np.square(pt_line_dist))

def plot_fundamental(ax, F, p1, p2, I):
    """
    Function to display epipolar lines and corresponding points
    param: F (3x3): fundamental matrix: (pt in frame 2).T * F * (pt in frame 1) = 0
    param: p1 (Nx2): 2d points on frame 1
    param: p2 (Nx2): 2d points on frame 2
    param: I: frame 2
    """
    N = p1.shape[0]
    P1 = np.c_[p1, np.ones((N,1))].transpose()
    P2 = np.c_[p2, np.ones((N,1))].transpose()
    L2 = np.matmul(F, P1).transpose() # transform points from 

    # the first image to get epipolar lines in the second image
    L2_norm = np.sqrt(L2[:,0]**2 + L2[:,1]**2)
    L2 = L2 / L2_norm[:,np.newaxis]
    pt_line_dist = np.multiply(L2, P2.T).sum(axis=1)
    closest_pt = p2 - (L2[:,0:2]*pt_line_dist[:,np.newaxis])

    # Find endpoints of segment on epipolar line (for display purposes).
    # offset from the closest point is 10 pixels
    pt1 = closest_pt - np.c_[L2[:,1], -L2[:,0]]*10 
    pt2 = closest_pt + np.c_[L2[:,1], -L2[:,0]]*10

    # Display points and segments of corresponding epipolar lines.
    # You will see points in red corsses, epipolar lines in green 
    # and a short cyan line that denotes the shortest distance between
    # the epipolar line and the corresponding point.
    ax.set_aspect('equal')
    ax.imshow(np.array(I))
    ax.plot(p2[:,0],p2[:,1],  '+r')
    ax.plot([p2[:,0], closest_pt[:,0]],[p2[:,1], closest_pt[:,1]], 'r')
    ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')


# write your code here for part estimating essential matrices
def fit_fundamental(matches):
    """
    Solves for the fundamental matrix using the matches with unnormalized method.
    """
    l = len(matches)
    M = np.zeros((l, 9))
    for i in range(l):
        u = matches[i][0]
        v = matches[i][1]
        u_2 = matches[i][2]
        v_2 = matches[i][3]
        M[i] = [u_2*u, u_2*v, u_2, v_2*u, v_2*v, v_2, u, v, 1]
    u, s, v = np.linalg.svd(M)
    F = v[-1]  
    u, s, v = np.linalg.svd(F.reshape(3,3))
    s, s[-1]= np.diag(s), 0
    F = np.matmul(u, np.matmul(s, v))
    return F

def fit_fundamental_normalized(matches):
    """
    Solve for the fundamental matrix using the matches with normalized method.
    """
    x = np.std(matches[:,0])
    x_1 = np.mean(matches[:,0:2], axis = 0)
    y = np.std(matches[:,1])
    z = np.std(matches[:,2])
    z_1 = np.mean(matches[:,2:4], axis = 0)
    k = np.std(matches[:,3])

    T1 = [[np.sqrt(2) / x, 0, -np.sqrt(2) / x * x_1[0]], 
          [0, np.sqrt(2) / y, -np.sqrt(2) / y * x_1[1]], 
          [0, 0, 1]]
    T2 = [[np.sqrt(2) / z, 0, -np.sqrt(2) / z * z_1[0]], 
          [0, np.sqrt(2) / k, -np.sqrt(2) / k * z_1[1]], 
          [0, 0, 1]]
    
    norm = np.zeros((len(matches), len(matches[0])))
    p1 = np.concatenate((matches[:,0:2], np.ones((len(matches),1))), axis = 1)
    p2 = np.concatenate((matches[:,2:4], np.ones((len(matches),1))), axis = 1)
    norm[:,0:2] = (np.matmul(T1, p1.T).T)[:,0:2]
    norm[:,2:4] = (np.matmul(T2, p2.T).T)[:,0:2]
    
    
    F = np.matmul(np.matmul(np.transpose(T2), fit_fundamental(norm)), T1)
    return F



# ---------------------------PART1-------------------------------------------------------------------------------------------------

# Fundamental matrix estimation
name = 'library' 
# You also need to report results for name = 'lab'
# name = 'lab'

I1 = Image.open('./{:s}1.jpg'.format(name))
I2 = Image.open('./{:s}2.jpg'.format(name))
matches = np.loadtxt('./{:s}_matches.txt'.format(name))
N = len(matches);

## Display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
I3 = np.zeros((I1.size[1],I1.size[0]*2,3))
I3[:,:I1.size[0],:] = I1;
I3[:,I1.size[0]:,:] = I2;
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(matches[:,0],matches[:,1],  '+r')
ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
ax.imshow(np.array(I3).astype(np.uint8))

# non-normalized method
F = fit_fundamental(matches) # <YOUR CODE>
pt1_2d = matches[:, :2]
pt2_2d = matches[:, 2:]
v2 = get_residual(F, pt1_2d, pt2_2d)
v1 = get_residual(F.T, pt2_2d, pt1_2d)
print('{:s}: residual in frame 2 (non-normalized method) = '.format(name), v2)
print('{:s}: residual in frame 1 (non-normalized method) = '.format(name), v1)
print('{:s}: residual combined   (non-normalized method) = '.format(name), (v1+v2)/2)
# Plot epipolar lines in image I2
fig, ax = plt.subplots()
plot_fundamental(ax, F, pt1_2d, pt2_2d, I2)
# Plot epipolar lines in image I1
fig, ax = plt.subplots()
plot_fundamental(ax, F.T, pt2_2d, pt1_2d, I1)
plt.show()

# normalized method
F = fit_fundamental_normalized(matches) # <YOUR CODE>
pt1_2d = matches[:, :2]
pt2_2d = matches[:, 2:]
v2 = get_residual(F, pt1_2d, pt2_2d)
v1 = get_residual(F.T, pt2_2d, pt1_2d)
print('{:s}: residual in frame 2 (normalized method) = '.format(name), v2)
print('{:s}: residual in frame 1 (normalized method) = '.format(name), v1)
print('{:s}: residual combined   (normalized method) = '.format(name), (v1+v2)/2)
# Plot epipolar lines in image I2
fig, ax = plt.subplots()
plot_fundamental(ax, F, pt1_2d, pt2_2d, I2)
# Plot epipolar lines in image I1
fig, ax = plt.subplots()
plot_fundamental(ax, F.T, pt2_2d, pt1_2d, I1)
plt.show()


# ----------------------------PART2-------------------------------------------------------------------------------------------------
print('Part 2: Camera Calibration')
def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

# Write your code here for camera calibration (lab)
def camera_calibration(pts_3d, pts_2d):
    """
    write your code to compute camera matrix
    """
    A = np.zeros((len(pts_3d) * 2, 12))
    for i in range(len(pts_3d)):
        temp = np.concatenate((pts_3d[i], [1]), axis = 0)
        A[2*i,4:8] = temp
        A[2*i,8:12] = -pts_2d[i, 1]*temp
        A[2*i+1,0:4] = temp
        A[2*i+1,8:12] = -pts_2d[i, 0]*temp
    u, s, v = np.linalg.svd(A)
    A = v[len(v)-1].reshape(3,4)
    return A


# Load 3D points, and their corresponding locations in 
# the two images.
pts_3d = np.loadtxt('./lab_3d.txt')
matches = np.loadtxt('./lab_matches.txt')

# <YOUR CODE> print lab camera projection matrices:
lab1_proj = camera_calibration(pts_3d, matches[:,0:2])
lab2_proj = camera_calibration(pts_3d, matches[:,2:4])
print('lab 1 camera projection')
print(lab1_proj)

print('')
print('lab 2 camera projection')
print(lab2_proj)

# <YOUR CODE> evaluate the residuals for both estimated cameras
_, lab1_res = evaluate_points(lab1_proj, matches[:,0:2], pts_3d)
print('residuals between the observed 2D points and the projected 3D points:')
print('residual in lab1:', lab1_res)
_, lab2_res = evaluate_points(lab2_proj, matches[:,2:4], pts_3d)
print('residual in lab2:', lab2_res)


lib1_proj = np.loadtxt('./library1_camera.txt')
lib2_proj = np.loadtxt('./library2_camera.txt')
print('library1 camera projection')
print(lib1_proj)
print('library2 camera projection')
print(lib2_proj)


# ----------------------------PART3-------------------------------------------------------------------------------------------------
print('Part 3: Camera Center')
# Write your code here for computing camera centers
def calc_camera_center(M):
    """
    write your code to get camera center in the world 
    from the projection matrix
    """
    # <YOUR CODE>
    u, s, v = np.linalg.svd(M)
    return v[-1] / v[-1][-1]

# <YOUR CODE> compute the camera centers using 
# the projection matrices
lab1_c = calc_camera_center(lab1_proj)
lab2_c = calc_camera_center(lab2_proj)
print('lab1 camera center', lab1_c)
print('lab2 camera center', lab2_c)

# <YOUR CODE> compute the camera centers with the projection matrices
lib1_c = calc_camera_center(lab1_proj)
lib2_c = calc_camera_center(lab2_proj)
print('library1 camera center', lib1_c)
print('library2 camera center', lib2_c)



# ----------------------------PART4-------------------------------------------------------------------------------------------------

print('Part 4: Triangulation')

def triangulation(proj1, proj2, matches):
    """
    write your code to triangulate the points in 3D
    """
    # <YOUR CODE>
    points_3d = np.zeros((len(matches), 4))
    for i in range(len(matches)):
        M1 = np.array([[0, -1, matches[i,1]], 
                          [1, 0, -matches[i,0]]]) 
            
        M2 = np.array([[0, -1, matches[i,3]], 
                          [1, 0, -matches[i,2]]])
        M = np.concatenate((np.matmul(M1, proj1), np.matmul(M2, proj2)))
        u, s, v = np.linalg.svd(M)
        points_3d[i] = v[len(v)-1]/v[len(v)-1][-1]

    return points_3d

def evaluate_points_3d(points_3d_gt, points_3d_lab) :
    """
    write your code to evaluate the triangulated 3D points
    """
    eval = np.sum((points_3d_lab[:, :-1] - points_3d_gt )**2)
    return eval

# triangulate the 3D point cloud for the lab data 
matches_lab = np.loadtxt('./lab_matches.txt')
points_3d_gt = np.loadtxt('./lab_3d.txt')
points_3d_lab = triangulation(lab1_proj, lab2_proj, matches_lab) # <YOUR CODE>
res_3d_lab = evaluate_points_3d(points_3d_gt, points_3d_lab) # <YOUR CODE>
print('Mean 3D reconstuction error for the lab data: ', round(np.mean(res_3d_lab), 5))
_, res_2d_lab1 = evaluate_points(lab1_proj, matches_lab[:,0:2], points_3d_lab[:,0:3])
_, res_2d_lab2 = evaluate_points(lab2_proj, matches_lab[:,2:4], points_3d_lab[:,0:3])
print('2D reprojection error for the lab 1 data: ', np.mean(res_2d_lab1))
print('2D reprojection error for the lab 2 data: ', np.mean(res_2d_lab2))
# visualization
camera_centers = np.vstack((lab1_c, lab2_c))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d_lab[:, 0], points_3d_lab[:, 1], points_3d_lab[:, 2], c='b', label='Points')
ax.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], c='g', s=50, marker='^', label='Camera Centers')
ax.legend(loc='best')
fig.savefig('lab_point_cloud.png')

# # triangulate the 3D point cloud for the library data
matches_lib = np.loadtxt('./library_matches.txt')
points_3d_lib = triangulation(lib1_proj, lib2_proj, matches_lib) # <YOUR CODE>
_, res_2d_lib1 = evaluate_points(lib1_proj, matches_lib[:,0:2], points_3d_lib[:,0:3])
_, res_2d_lib2 = evaluate_points(lib2_proj, matches_lib[:,2:4], points_3d_lib[:,0:3])
print('2D reprojection error for the library 1 data: ', np.mean(res_2d_lib1))
print('2D reprojection error for the library 2 data: ', np.mean(res_2d_lib2))
# visualization
camera_centers_library = np.vstack((lib1_c, lib2_c))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d_lib[:, 0], points_3d_lib[:, 1], points_3d_lib[:, 2], c='b', label='Points')
ax.scatter(camera_centers_library[:, 0], camera_centers_library[:, 1], 
           camera_centers_library[:, 2], c='g', s=90, 
           marker='^', label='Camera Centers')
ax.view_init(azim=-45, elev=45)
ax.legend(loc='best')
fig.savefig('library_point_cloud.png')