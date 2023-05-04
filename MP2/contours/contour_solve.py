import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import cv2

def non_maximum_suppression(img, theta):
    Z = np.zeros(img.shape)
    h, w = Z.shape
    theta = theta * 180. / np.pi
    theta[theta < 0] += 180
    for i in range(1,h-1):
        for j in range(1,w-1):
            if (theta[i,j] < 22.5) or (theta[i,j] >= 157.5):
                q = img[i, j+1]
                r = img[i, j-1]

            elif (22.5 <= theta[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]

            elif (67.5 <= theta[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]

            elif (112.5 <= theta[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0
    return Z/Z.max()*255

def compute_edges_dxdy(I):
  """Returns the norm of dx and dy as the edge response function."""
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  I = I.astype(np.float32)/255.
  #q1 Warm-up
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')

  #q2 Smoothing
  #dx
  dx_I = gaussian_filter(I, 2)
  derivative = np.array([[-1, 0, 1]])
  dx_I = signal.convolve2d(dx_I, derivative, mode='same', boundary='symm')
  dx = dx_I
  #dy
  dy_I = gaussian_filter(I, 2)
  derivative = np.array([[-1, 0, 1]]).T
  dy_I = signal.convolve2d(dy_I, derivative, mode='same', boundary='symm')
  dy = dy_I

  #q3 Non-maximum Suppression:
  mag = np.sqrt(dx**2 + dy**2)
  theta = np.arctan2(dy, dx)
  mag = non_maximum_suppression(mag, theta)

  mag = mag / mag.max() * 255
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)
  return mag
