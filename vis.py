import matplotlib.pyplot as plt
import torch
import numpy as np

du = torch.load('./output/lego_undistorted/u_distortion30000.pt')
dv = torch.load('./output/lego_undistorted/v_distortion30000.pt')
tensor = 300 * torch.stack((du, dv), dim=2).detach().cpu()
l_x, l_y, _ = tensor.shape

subsample_factor = 10
x, y = np.meshgrid(np.arange(0, l_x, subsample_factor), np.arange(0, l_y, subsample_factor))
dx = tensor[::subsample_factor, ::subsample_factor, 0].numpy()
dy = tensor[::subsample_factor, ::subsample_factor, 1].numpy()

# Visualization
plt.figure(figsize=(10, 10))
plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
plt.xlim([0, l_x])
plt.ylim([0, l_y])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Vector Directions on Subsampled Grid')

# Invert the y-axis to match the conventional origin location for images
plt.gca().invert_yaxis()

plt.show()

#distortion_params = torch.load('./output/lego_undistorted/distortion_params.pt').detach().cpu()
#height, width = 200, 200
#y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
#points = torch.stack((x.flatten(), y.flatten()), dim=-1)
#
## Apply distortion to each point
#def apply_distortion(points, distortion_params):
#    x = points[:, 0]
#    y = points[:, 1]
#    k1, k2, k3, k4, k5, k6, p1, p2 = distortion_params
#
#    x2 = x * x
#    y2 = y * y
#    r2 = x2 + y2
#    _2xy = 2 * x * y
#
#    # If r2 is greater than 2, it is considered too far and should be ignored in this context
#    # Here we'll just set a flag for these points
#    to_ignore = r2 > 2
#
#    radial_u = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
#    radial_v = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
#    radial = radial_u / radial_v
#
#    tangential_x = p1 * _2xy + p2 * (r2 + 2 * x2)
#    tangential_y = p1 * (r2 + 2 * y2) + p2 * _2xy
#
#    x_d = x * radial + tangential_x
#    y_d = y * radial + tangential_y
#
#    distorted_points = torch.stack((x_d, y_d), dim=-1)
#
#    return distorted_points, to_ignore
#
## Apply the distortion
#distorted_points, to_ignore = apply_distortion(points, distortion_params)
#
## Reshape for visualization
#distorted_points = distorted_points.reshape(height, width, 2)
#to_ignore = to_ignore.reshape(height, width)
#points = points.reshape(height, width, 2)
#tensor = 50 * (distorted_points - points)
#l_x, l_y, _ = tensor.shape
#
#subsample_factor = 10
#x, y = np.meshgrid(np.arange(0, l_x, subsample_factor), np.arange(0, l_y, subsample_factor))
#dx = tensor[::subsample_factor, ::subsample_factor, 0].numpy()
#dy = tensor[::subsample_factor, ::subsample_factor, 1].numpy()
#
## Visualization
#plt.figure(figsize=(10, 10))
#plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
#plt.xlim([0, l_x])
#plt.ylim([0, l_y])
#plt.xlabel('X axis')
#plt.ylabel('Y axis')
#plt.title('Vector Directions on Subsampled Grid')
#
## Invert the y-axis to match the conventional origin location for images
#plt.gca().invert_yaxis()
#
#plt.show()
