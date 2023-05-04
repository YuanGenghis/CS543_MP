import numpy as np
from scipy.ndimage import gaussian_filter

def blend(im1, im2, mask):
  mask = mask / 255.

  # generate Gaussian pyramid and Laplacian Pyramid for A
  G = im1.copy()
  gpA = [G/255.]
  lpA = []
  
  # generate Gaussian pyramid and Laplacian Pyramid for B
  G = im2.copy()
  gpB = [G/255.]
  lpB = []

  # generate Gaussian pyramid and Laplacian Pyramid for mask
  G = mask.copy()
  gpM = [G]

  lr = []
  for i in range(6):
    gpA.append(gaussian_filter(gpA[-1], 2**i))
    lpA.append(gpA[-2] - gpA[-1])
    gpB.append(gaussian_filter(gpB[-1], 2**i))
    lpB.append(gpB[-2] - gpB[-1])
    gpM.append(gaussian_filter(gpM[-1], 2**i))
    lr.append(lpA[-1] * gpM[-2] + lpB[-1]*(1-gpM[-2]))
  

  g = gpA[-1] * gpM[-1] + gpB[-1] * (1 - gpM[-1])

  real = sum(lr) + g

  return real * 255
