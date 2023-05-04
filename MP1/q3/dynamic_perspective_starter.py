import numpy as np
import matplotlib.pyplot as plt

def get_wall_z_image(Z_val, fx, fy, cx, cy, szx, szy):
    Z = Z_val*np.ones((szy, szx), dtype=np.float32)
    return Z


def get_road_z_image(H_val, fx, fy, cx, cy, szx, szy):
    y = np.arange(szy).reshape(-1,1)*1.
    y = np.tile(y, (1, szx))
    Z = np.zeros((szy, szx), dtype=np.float32)
    Z[y > cy] = H_val*fy / (y[y>cy]-cy)
    Z[y <= cy] = np.NaN
    return Z


def plot_optical_flow(ax, Z, u, v, cx, cy, szx, szy, s=16):
    # Here is a function for plotting the optical flow. Feel free to modify this 
    # function to work well with your inputs, for example if your predictions are
    # in a different coordinate frame, etc.

    x, y = np.meshgrid(np.arange(szx), np.arange(szy))
    ax.imshow(Z, alpha=0.5, origin='upper')
    # ax.quiver is to used to plot a 2D field of arrows. 
    # please check https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html to
    # understand the detials of the plotting with ax.quiver
    q = ax.quiver(x[::s,::s], y[::s,::s], u[::s,::s], v[::s, ::s], angles='xy')
    ax.axvline(cx)
    ax.axhline(cy)
    ax.set_xlim([0, szx])
    ax.set_ylim([szy, 0])
    ax.axis('equal')

# if __name__ == "__main__":
#     # Focal length along X and Y axis. In class we assumed the same focal length 
#     # for X and Y axis. but in general they could be different. We are denoting 
#     # these by fx and fy, and assume that they are the same for the purpose of
#     # this MP.
#     fx = fy = 128.

#     # Size of the image
#     szy = 256
#     szx = 384   

#     # Center of the image. We are going to assume that the principal point is at 
#     # the center of the image.
#     cx = 192
#     cy = 128

#     # Gets the image of a wall 2m in front of the camera.
#     Z1 = get_wall_z_image(2., fx, fy, cx, cy, szx, szy)


#     # Gets the image of the ground plane that is 3m below the camera.
#     Z2 = get_road_z_image(3., fx, fy, cx, cy, szx, szy)

#     fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
#     ax1.imshow(Z1)
#     ax2.imshow(Z2)
    
#     # Plotting function.
#     f = plt.figure(figsize=(13.5,9))
#     u = np.random.rand(*Z1.shape)
#     v = np.random.rand(*Z1.shape)

#     #3.1
#     # calculate the optical flow at each column of the image.
#     for x in range(szx):
#         for y in range(szy):
#             # Calculating the optical flow vector.
#             Z = Z2[y, x]
#             if Z > 0:
#                 u[y, x] = fx * (x - cx) / Z
#                 v[y, x] = fy * (y - cy) / Z

#     #3.2
#     Z = get_road_z_image(5., fx, fy, cx, cy, szx, szy)
#     # Calculating the x component of the flow vectors
#     u2 = np.ones(Z.shape)

#     # Calculating the y component of the flow vectors
#     v2 = np.zeros(Z.shape)
#     plot_optical_flow(f.gca(), Z, u2, v2, cx, cy, szx, szy, s=16)

#     # plot_optical_flow(f.gca(), Z2, u, v, cx, cy, szx, szy, s=16)
#     f.savefig('optical_flow_output_3_2.png', bbox_inches='tight')

## 3.1-----------------------------
# if __name__ == "__main__":
#     fx = fy = 128.

#     # Size of the image
#     szy = 256
#     szx = 384

#     # Center of the image. We are going to assume that the principal point is at 
#     # the center of the image.
#     cx = 192
#     cy = 128

#     # Gets the image of a wall 2m in front of the camera.
#     Z1 = get_wall_z_image(2., fx, fy, cx, cy, szx, szy)

#     # Gets the image of the ground plane that is 3m below the camera.
#     Z2 = get_road_z_image(3., fx, fy, cx, cy, szx, szy)

#     fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
#     ax1.imshow(Z1)
#     ax2.imshow(Z2)
    
#     # Plotting function.
#     f = plt.figure(figsize=(13.5,9))
#     u = np.zeros(Z2.shape)
#     v = np.zeros(Z2.shape)
#     for x in range(szx):
#         for y in range(szy):
#             # Calculating the optical flow vector.
#             Z = Z2[y, x]
#             if Z > 0:
#                 u[y, x] = fx * (x - cx) / Z
#                 v[y, x] = fy * (y - cy) / Z
        
#     plot_optical_flow(f.gca(), Z2, u, v, cx, cy, szx, szy, s=16)
#     f.savefig('optical_flow_output.png', bbox_inches='tight')

## 3.2-----------------------------
#Sitting in a train and looking out over a flat field from a side window.
# if __name__ == "__main__":
#     fx = fy = 128.

#     szy = 256
#     szx = 384

#     cx = 192
#     cy = 128

#     Z = get_road_z_image(128., fx, fy, cx, cy, szx, szy)


#     # Calculating the x component of the flow vectors
#     u = np.ones(Z.shape)

#     # Calculating the y component of the flow vectors
#     v = np.zeros(Z.shape)

#     # Plotting the induced optical flow
#     f = plt.figure(figsize=(13.5,9))
#     plot_optical_flow(f.gca(), Z, u, v, cx, cy, szx, szy, s=16)
#     f.savefig('optical_flow_output.png', bbox_inches='tight')


# 3.3-----------------------------
if __name__ == "__main__":
    fx = fy = 128.
    szx = 384
    szy = 256
    cx = 192
    cy = 128

    # Image of the wall in front of the camera
    Z3 = get_wall_z_image(2., fx, fy, cx, cy, szx, szy)

    # Induced optical flow for scenario 3
    u = np.zeros(Z3.shape)
    v = np.zeros(Z3.shape)

    x, y = np.meshgrid(np.arange(szx), np.arange(szy))
    y = (y - cy) / fy
    x = (x - cx) / fx

    u = x / Z3
    v = y / Z3

    # Plotting the induced optical flow for scenario 3
    f = plt.figure(figsize=(13.5,9))
    plot_optical_flow(f.gca(), Z3, u, v, cx, cy, szx, szy, s=16)
    f.savefig('scenario_3_output.png', bbox_inches='tight')

##3.5----------------------------------------
# if __name__ == "__main__":
#     fx = fy = 128.

#     szy = 256
#     szx = 384

#     cx = 192
#     cy = 128

#     Z3 = get_wall_z_image(2., fx, fy, cx, cy, szx, szy)

#     # Induced optical flow for scenario 5
#     u = np.zeros(Z3.shape)
#     v = np.zeros(Z3.shape)

#     x, y = np.meshgrid(np.arange(szx), np.arange(szy))
#     y = (y - cy) / fy
#     x = (x - cx) / fx

#     r = np.sqrt(x**2 + y**2)
#     theta = np.arctan2(y, x)

#     u = r * np.sin(theta) / Z3
#     v = r * np.cos(theta) / Z3

#     # Plotting the induced optical flow for scenario 5
#     f = plt.figure(figsize=(13.5,9))
#     plot_optical_flow(f.gca(), Z3, -u, -v, cx, cy, szx, szy, s=16)
#     f.savefig('scenario_5_output.png', bbox_inches='tight')
