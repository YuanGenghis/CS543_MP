import numpy as np
import scipy
import skimage
import skimage.io
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
from heapq import *
import itertools
import random
from skimage.transform import *

def imread(fname):
    """
    read image into np array from file
    """
    return skimage.io.imread(fname)

def imread_bw(fname):
    """
    read image as gray scale format
    """
    return cv2.cvtColor(imread(fname), cv2.COLOR_BGR2GRAY)

def imshow(img):
    """
    show image
    """
    skimage.io.imshow(img)
    
def get_sift_data(img):
    """
    detect the keypoints and compute their SIFT descriptors with opencv library
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    plot the match between two image according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')

def plot_panorama(ax, img):
    ax.set_aspect('equal')
    ax.imshow(img, cmap='gray')
    ax.axis('off')



def get_best_matches(img1, img2, num_matches):
    kp1, des1 = get_sift_data(img1)
    kp2, des2 = get_sift_data(img2)
    kp1, kp2 = np.array(kp1), np.array(kp2)
    
    # Find distance between descriptors in images
    dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')
    
    # Write your code to get the matches according to dist
    # <YOUR CODE>

    n1, n2 = dist.shape
    matches = []
    dist_f = dist.reshape(-1,)
    idx = np.argpartition(dist_f, num_matches)
    for i in idx[:num_matches]:
        h = i // n2
        j = i % n2
        x1, y1 = kp1[h].pt
        x2, y2 = kp2[j].pt
        matches.append([x1, y1, x2, y2])

    return np.array(matches)


def ransac(data, threshold=10, iterations=10):
    """
    write your ransac code to find the best model, inliers, and residuals
    """
    # <YOUR CODE>
    # we have img1 and img2 together with the coordincate of their matches
    # now we want to find a transformation that have most number of inliers
    # for each computation:
    # we need to find four random points from matches, find the transformation and find the number of inliers
    num_matches, _ = data.shape
    max_inlier = 0
    best_T = None
    new_data = None
    residue = 0
    cur_pairs_list = [x for x in range(num_matches)]
    for it in range(iterations):
        A = None
        random.shuffle(cur_pairs_list)
        # construct the homography function
        # Ah = 0, A has size 8 x 9 and h has size 9 x 1
        # A is known and h is unknown and is the transformation we want
        # A = [0.T,    x_i.T, -y_i'*x_i.T]
        #     [x_i.T,    0.T, -x_i'*x_i.T]
        # i from 0 to 3 (we pick 4 points per iteration)
        # pair_i = data[i, :]
        for i in range(4):
            x1, y1, x2, y2 = data[cur_pairs_list[i], :]
            if i == 0:
                A = np.array([[0,   0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2],
                              [x1, y1, 1,  0,  0, 0, -x2*x1, -x2*y1, -x2]])
            else:
                new_A = np.array([[0,   0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2],
                                  [x1, y1, 1,  0,  0, 0, -x2*x1, -x2*y1, -x2]])
                A = np.vstack((A, new_A))
        
        # solve for linear equation
        u, s, vh = np.linalg.svd(A)
        h = vh[-1:].reshape(3, 3)
        # h /= h[2][2]
        if np.linalg.matrix_rank(h) < 3: 
            continue
        # after solving h, we need to find the inlier numbers of all matches pairs
        # ground truth (x2, y2) are data[:, 2:4]
        # estimate (x2, y2) are h @ data[:, 0:2].T (append 1 to the last row)
        data_from_1 = np.vstack((data[:, 0:2].T, np.ones((1, num_matches))))
        estimate_data = h @ data_from_1
        estimate_data = estimate_data[0:2, :] / estimate_data[2:3, :]
        # find the distance between estimate_data and ground truth
        diff = np.linalg.norm(estimate_data - data[:, 2:4].T, axis=0)**2
        inlier_residue = list(filter(lambda x: (x < threshold), diff))
        inlier_count = len(inlier_residue)
        # update best_T and count
        if inlier_count > max_inlier:
            max_inlier = inlier_count
            best_T = h
            residue = inlier_residue
            new_data = data[np.where(diff < threshold)]
            
    return best_T, new_data, max_inlier, residue


def compute_homography(img1, img2, h):
    """
    write your code to compute homography according to the matches
    """
    # <YOUR CODE>
    # h is the transformation if fixed img2, how img1 is transformed
    # first find the range of image indices
    height1, width1 = img1.shape[0:2]
    height2, width2 = img2.shape[0:2]
    corners1 = np.array([[0, 0, 1], [0, height1-1, 1], [width1-1, height1-1, 1], [width1-1, 0, 1]]).T
    corners2 = np.array([[0, 0, 1], [0, height2-1, 1], [width2-1, height2-1, 1], [width2-1, 0, 1]]).T
    new_corners1 = h @ corners1
    new_corners1 = new_corners1[0:2, :] / new_corners1[2:3, :]
    # find the range of the new_corners
    xmin1, xmax1 = np.min(new_corners1[0, :]), np.max(new_corners1[0, :])
    ymin1, ymax1 = np.min(new_corners1[1, :]), np.max(new_corners1[1, :])
    # now the image should range from xmin to width2, from ymin to height2
    yrange = np.ceil(np.maximum(height2, ymax1) - np.minimum(ymin1, height2))
    xrange = np.ceil(np.maximum(width2, xmax1) - np.minimum(xmin1, height2))
    # now we know we need to do a translation to the image
    xshift = -xmin1
    yshift = -ymin1
    affine = np.array([[1, 0, xshift],
                       [0, 1, yshift],
                       [0, 0,      1]])
    h1 = affine @ h
    h2 = affine
      
    img1_warped = warp(img1, np.linalg.inv(h1), output_shape=(yrange, xrange))
    img2_warped = warp(img2, np.linalg.inv(h2), output_shape=(yrange, xrange))

    fig1, ax1 = plt.subplots(figsize=(20,10))
    plot_panorama(ax1, img1_warped)
    fig2, ax2 = plt.subplots(figsize=(20,10))
    plot_panorama(ax2, img2_warped)
    return img1_warped, img2_warped
    

def warp_images(img1, img2):
    """
    write your code to stitch images together according to the homography
    """
    # <YOUR CODE>
    # we are given two images, we need to warp them together
    # assume we are given color images, which have 3 color channels
    # img1 and img2 should have the same size
    height, width = img1.shape[0:2]
    warp_image = np.zeros_like(img1)
    for i in range(height):
        for j in range(width):
            for c in range(3):
                if (img1[i, j, c] == 0):
                    warp_image[i, j, c] = img2[i, j, c]
                elif (img2[i, j, c] == 0):
                    warp_image[i, j, c] = img1[i, j, c]
                else:
                    warp_image[i, j, c] = 0.5 * (img1[i, j, c] + img2[i, j, c])
    
    fig, ax = plt.subplots(figsize=(20,10))
    plot_panorama(ax, warp_image)
    return warp_image


img1 = imread('./stitch/left.jpg')
img2 = imread('./stitch/right.jpg')


data = get_best_matches(img1, img2, 300)
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, img1, img2, data)
fig.savefig('sift_match.pdf', bbox_inches='tight')

# display the inlier matching, report the average residual
best_T, new_data, max_inlier, residue = ransac(data, threshold=5, iterations=5000)
print(best_T)
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, img1, img2, new_data)
print("Average residual:", np.average(residue))
print("Inliers:", max_inlier)
fig.savefig('ransac_match.pdf', bbox_inches='tight')
img1_warped, img2_warped = compute_homography(img1, img2, best_T)
cv2.imwrite('stitched_images1.jpg', img1_warped[:,:,::-1]*255., 
            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
cv2.imwrite('stitched_images2.jpg', img2_warped[:,:,::-1]*255., 
            [int(cv2.IMWRITE_JPEG_QUALITY), 90])


warp_image = warp_images(img1_warped, img2_warped)
# part (e) warp images to stitch them together, 
# display and report the stitching results
# <YOUR CODE>
cv2.imwrite('stitched_images.jpg', warp_image[:,:,::-1]*255., 
            [int(cv2.IMWRITE_JPEG_QUALITY), 90])