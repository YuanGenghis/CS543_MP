import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
from copy import copy, deepcopy
import scipy

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



def get_best_matches(img1, img2, num_matches):
    kp1, des1 = get_sift_data(img1)
    kp2, des2 = get_sift_data(img2)
    kp1, kp2 = np.array(kp1), np.array(kp2)
    
    # Find distance between descriptors in images
    dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')
    data = []
    temp = deepcopy(dist)
    dist = np.ravel(dist)
    dist.sort() 
    for i in range(num_matches):
        k, q = np.where(temp == dist[i])
        kp1_0 = (int(kp1[k[0]].pt[0]))
        kp1_1 = (int(kp1[k[0]].pt[1]))
        kp2_0 = (int(kp2[q[0]].pt[0]))
        kp2_1 = (int(kp2[q[0]].pt[1]))
        data.append((kp1_0, kp1_1, kp2_0, kp2_1))

    return np.array(data)

def ransac(data):
    """
    write your ransac code to find the best model, inliers, and residuals
    """
    # <YOUR CODE>
    best_inliers_num = 0
    thres = 5
    for ite in range(400):
        matches = []
        for i in range(4):
            k = np.random.randint(0, len(data))
            matches.append(data[k])
        H = compute_homography(matches)
        if np.linalg.matrix_rank(H) < 3:
            continue
        one_col = np.ones((len(data),1))
        left = np.concatenate((data[:,0:2], one_col), axis = 1)
        right = np.concatenate((data[:,2:4], one_col), axis = 1)
        inliers = []
        total_error = 0
        for i in range(len(data)):
            temp = np.matmul(H, left[i])
            temp = temp/temp[-1]
            error = np.linalg.norm(temp - right[i])**2
            if error < thres:
                tmp = np.concatenate((left[i][0:2], right[i][0:2])).tolist()
                inliers.append(tmp)
                total_error += error
        inliers_num = len(inliers)
        if inliers_num > best_inliers_num:
            max_inliers = inliers.copy()
            best_inliers_num = inliers_num
            best_H = H
            best_model_errors = total_error / best_inliers_num

    return max_inliers, best_model_errors, best_H

def compute_homography(matches):
    """
    write your code to compute homography according to the matches
    """
    # <YOUR CODE>
    A = np.zeros((8,9))
    for i in range(4):
        m_0 = matches[i][0]
        m_1 = matches[i][1]
        m_2 = matches[i][2]
        m_3 = matches[i][3]
        A[i*2,:] = [0, 0, 0, m_0, m_1, 1, -m_3 * m_0, -m_3 * m_1, -m_3*1]
        A[i*2+1,:] = [m_0, m_1, 1, 0, 0, 0, -m_2 * m_0, -m_2 * m_1, -m_2*1]
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))
    return H

def warp_images(img1, img2, H):
    #write your code to stitch images together according to the homography
    canvas = (img1.shape[1] + img2.shape[1], img1.shape[0] + img2.shape[0])
    img2_T = np.array([[1, 0, img1.shape[1]], [0, 1, 0], [0, 0, 1]]).astype(float)

    projected_img1 = cv2.warpPerspective(img1, img2_T @ H, canvas)
    projected_img2 = cv2.warpPerspective(img2, img2_T, canvas)
    
    idx = (np.sum(projected_img2 == 0, axis=2) == 3)
    canvas = projected_img2.copy()
    canvas[idx,:] = projected_img1[idx,:]
    
    return canvas



img1 = imread('./stitch/left.jpg')
img2 = imread('./stitch/right.jpg')


data = get_best_matches(img1, img2, 300)
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, img1, img2, data)
fig.savefig('sift_match.pdf', bbox_inches='tight')

# display the inlier matching, report the average residual
# <YOUR CODE>
max_inliers, best_model_errors, best_H = ransac(data)
print("Average residual:", np.average(best_model_errors))
print("Inliers:", len(max_inliers))
plot_inlier_matches(ax, img1, img2, np.asarray(max_inliers))
fig.savefig('ransac_match.pdf', bbox_inches='tight')

# display and report the stitching results
output = warp_images(img1, img2, best_H)
plt.figure()
plt.imshow(output)
plt.savefig('./result/part1_stitched_colorimg.jpg')
plt.show()
cv2.imwrite('stitched_images.jpg', output[:,:,::-1], 
            [int(cv2.IMWRITE_JPEG_QUALITY), 90])