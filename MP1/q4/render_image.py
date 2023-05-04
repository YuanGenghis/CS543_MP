import numpy as np
from generate_scene import get_ball
import matplotlib.pyplot as plt

# specular exponent
k_e = 50

def render(Z, N, A, S, 
           point_light_loc, point_light_strength, 
           directional_light_dirn, directional_light_strength,
           ambient_light, k_e):
  # To render the images you will need the camera parameters, you can assume
  # the following parameters. (cx, cy) denote the center of the image (point
  # where the optical axis intersects with the image, f is the focal length.
  # These parameters along with the depth image will be useful for you to
  # estimate the 3D points on the surface of the sphere for computing the
  # angles between the different directions.
  h, w = A.shape
  cx, cy = w / 2, h /2
  f = 128.


  # Ambient Term
  I = A * ambient_light


  #Point light for both term and directoin light for specular term
  for i in range(h):
        for j in range(w):
              phsical_z = Z[i, j]
              x = (j - cx) * phsical_z / f
              y = (i - cy) * phsical_z / f
              Vi = point_light_loc[0] - np.array([x, y, Z[i, j]])
              Vi = Vi / np.linalg.norm(Vi)
              # Diffuse Term point light
              I[i, j] += np.maximum(np.dot(N[i, j], Vi), 0) * point_light_strength[0] * A[i, j]


              Si = 2 * np.dot(N[i, j], Vi) * N[i, j] + (-Vi)
              Si = Si / np.linalg.norm(Si)

              Vr = np.array([0, 0, 0]) - np.array([x, y, Z[i][j]])
              Vr = Vr / np.linalg.norm(Vr)

              # Specular Term point light
              specular_point_light = np.power(np.maximum(np.dot(Vr, Si), 0), k_e) * point_light_strength[0] * S[i, j]
              I[i, j] += specular_point_light

              directional_dir = np.array(directional_light_dirn[0])
              directional_dir = directional_dir / np.linalg.norm(directional_dir)

              dir_Si = - directional_dir - np.dot(2 * (np.dot( - directional_dir, N[i, j])),  N[i, j])
      
              dir_Si = dir_Si / np.linalg.norm(dir_Si)
              specular_directional_light = np.power(np.maximum(np.dot(Vr, dir_Si), 0), k_e) * directional_light_strength[0] * S[i, j]

              # Specular Term direction light
              I[i, j] += specular_directional_light
            
  # Diffuse Term direction light
  diffuse_directional_light = 0.0
  for i in range(len(directional_light_dirn)):
        directional_light_dir = np.array(directional_light_dirn[i])
        directional_light_dir = directional_light_dir / np.linalg.norm(directional_light_dir)
        diffuse_directional_light += np.maximum(np.dot(N, directional_light_dir), 0) * directional_light_strength[i]
  I += diffuse_directional_light * A

  I = np.minimum(I, 1)*255
  I = I.astype(np.uint8)
  I = np.repeat(I[:,:,np.newaxis], 3, axis=2)
  return I

def main():
  for specular in [True, False]:
    # get_ball function returns:
    # - Z (depth image: distance to scene point from camera center, along the
    # Z-axis)
    # - N is the per pixel surface normals (N[:,:,0] component along X-axis
    # (pointing right), N[:,:,1] component along Y-axis (pointing down),
    # N[:,:,2] component along Z-axis (pointing into the scene)),
    # - A is the per pixel ambient and diffuse reflection coefficient per pixel,
    # - S is the per pixel specular reflection coefficient.
    Z, N, A, S = get_ball(specular=specular)

    # Strength of the ambient light.
    ambient_light = 0.5
    
    # For the following code, you can assume that the point sources are located
    # at point_light_loc and have a strength of point_light_strength. For the
    # directional light sources, you can assume that the light is coming _from_
    # the direction indicated by directional_light_dirn (i.e., \hat{v}_i =
    # directional_light_dirn), and with strength directional_light_strength.
    # The coordinate frame is centered at the camera, X axis points to the
    # right, Y-axis point down, and Z-axis points into the scene.
    
    # Case I: No directional light, only point light source that moves around
    # the object. 
    point_light_strength = [1.5]
    directional_light_dirn = [[1, 0, 0]]
    directional_light_strength = [0.0]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      point_light_loc = [[10*np.cos(theta), 10*np.sin(theta), -3]]
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e)
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_point.png', bbox_inches='tight')
    plt.close()

    # Case II: No point source, just a directional light source that moves
    # around the object.
    point_light_loc = [[0, -10, 2]]
    point_light_strength = [0.0]
    directional_light_strength = [2.5]
    
    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    for theta in np.linspace(0, np.pi*2, 16): 
      directional_light_dirn = [np.array([np.cos(theta), np.sin(theta), .1])]
      directional_light_dirn[0] = \
        directional_light_dirn[0] / np.linalg.norm(directional_light_dirn[0])
      I = render(Z, N, A, S, point_light_loc, point_light_strength, 
                 directional_light_dirn, directional_light_strength,
                 ambient_light, k_e) 
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
    plt.savefig(f'specular{specular:d}_move_direction.png', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
  main()
