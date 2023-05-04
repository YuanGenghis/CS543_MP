import numpy as np
from scipy import signal
import scipy
import cv2

def compute_corners(I):
  # Currently this code proudces a dummy corners and a dummy corner response
  # map, just to illustrate how the code works. Your task will be to fill this
  # in with code that actually implements the Harris corner detector. You
  # should return th ecorner response map, and the non-max suppressed corners.
  # Input:
  #   I: input image, H x W x 3 BGR image
  # Output:
  #   response: H x W response map in uint8 format
  #   corners: H x W map in uint8 format _after_ non-max suppression. Each
  #   pixel stores the score for being a corner. Non-max suppressed pixels
  #   should have a low / zero-score.

  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  
  I = I.astype(np.float32)/255
  
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary = 'symm')
  Ixx, Iyy, Ixy = dx**2, dy**2, dx*dy
  Ix = scipy.ndimage.gaussian_filter(Ixx, 0.618, order=0, output=None, mode='reflect')
  Iy = scipy.ndimage.gaussian_filter(Iyy, 0.618, order=0, output=None, mode='reflect')
  xy = scipy.ndimage.gaussian_filter(Ixy, 0.618, order=0, output=None, mode='reflect')
  H, W = I.shape
  alpha = 0.06
  response = Ix*Iy - xy**2 - alpha*(Ix + Iy)**2

  corners = response / response.max() * 255
  for h in range(H):
      for w in range(W):
          for i in range(max(0, h - 4), min(h + 4, H)):
              for j in range(max(0, w - 4), min(w + 4, W)):
                  if response[i, j] > response[h, w]:
                      corners[h, w] = 0

  corners = corners.astype(np.uint8)
  
  response = response * 255.
  response = np.clip(response, 0, 255)
  response = response.astype(np.uint8)
  
  return response, corners
