import matplotlib.pyplot as plt
import torch
import numpy as np


du = torch.load('/home/youming/Desktop/playaround_gaussian_platting/output/test/u_distortion7000.pt').cpu().detach().numpy()
dv = torch.load('/home/youming/Desktop/playaround_gaussian_platting/output/test/v_distortion7000.pt').cpu().detach().numpy()
su = torch.load('/home/youming/Desktop/playaround_gaussian_platting/output/test/u_radial7000.pt').cpu().detach().numpy()
sv = torch.load('/home/youming/Desktop/playaround_gaussian_platting/output/test/v_radial7000.pt').cpu().detach().numpy()
height, width = 400, 400
y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
points = torch.stack((x.flatten(), y.flatten()), dim=-1)

# Apply distortion to each point
def apply_distortion(points, du, dv, su, sv):
    x = points[:, 0]
    y = points[:, 1]
    x_d = x * su + du
    y_d = y * sv + dv

    distorted_points = torch.stack((x_d, y_d), dim=-1)

    return distorted_points

# Apply the distortion
distorted_points = apply_distortion(points, du.flatten(), dv.flatten(), su.flatten(), sv.flatten())

# Reshape for visualization
distorted_points = distorted_points.reshape(height, width, 2)
points = points.reshape(height, width, 2)
tensor = 500 * (distorted_points - points)
l_x, l_y, _ = tensor.shape

subsample_factor = 2
x, y = np.meshgrid(np.arange(0, l_x, subsample_factor), np.arange(0, l_y, subsample_factor))
dx = tensor[::subsample_factor, ::subsample_factor, 0].numpy()
dy = tensor[::subsample_factor, ::subsample_factor, 1].numpy()

# Visualization
plt.figure(figsize=(25, 25))
plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
plt.xlim([0, l_x])
plt.ylim([0, l_y])
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Vector Directions on Subsampled Grid')

# Invert the y-axis to match the conventional origin location for images
plt.gca().invert_yaxis()

plt.show()



#distortion_params = torch.load('./output/test/radial7000.pt').detach().cpu()
#height, width = 200, 200
#y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
#points = torch.stack((x.flatten(), y.flatten()), dim=-1)
#
## Apply distortion to each point
#def apply_distortion(points, distortion_params):
#    x = points[:, 0]
#    y = points[:, 1]
#
#    x2 = x * x
#    y2 = y * y
#    r2 = x2 + y2
#    r = torch.sqrt(r2)
#    theta = torch.atan(r)
#    idx = (theta * 1000).long()
#    radial = distortion_params[idx]
#
#    x_d = x * radial
#    y_d = y * radial
#
#    distorted_points = torch.stack((x_d, y_d), dim=-1)
#
#    return distorted_points
#
## Apply the distortion
#distorted_points = apply_distortion(points, distortion_params)
#
## Reshape for visualization
#distorted_points = distorted_points.reshape(height, width, 2)
#points = points.reshape(height, width, 2)
#tensor = 100 * (distorted_points - points)
#l_x, l_y, _ = tensor.shape
#
#subsample_factor = 1
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



#distortion_params = torch.load('./output/lego_undistorted_8params/distortion_params.pt').detach().cpu()
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




#distortion_params = torch.load('/home/youming/Desktop/playaround_gaussian_platting/output/cube_global/poly_coeff7000.pt').detach().cpu()
##distortion_params = torch.tensor([0.01578328428387751, -0.027082591106441487, -0.002714465463926838, -0.00013107686083776261])
##distortion_params = torch.tensor([0, 0, 0, 0])
#height, width = 400, 400
#y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
#points = torch.stack((x.flatten(), y.flatten()), dim=-1)
#
## Apply distortion to each point
#def apply_distortion(points, distortion_params):
#    x = points[:, 0]
#    y = points[:, 1]
#    k1, k2, k3, k4 = distortion_params
#
#    inv_r = 1 / torch.sqrt(x**2 + y**2)
#    theta = torch.atan(torch.sqrt(x**2 + y**2))
#    rho = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
#
#    x_d = rho * inv_r * x
#    y_d = rho * inv_r * y
#
#    distorted_points = torch.stack((x_d, y_d), dim=-1)
#
#    return distorted_points
#
## Apply the distortion
#distorted_points = apply_distortion(points, distortion_params)
#
## Reshape for visualization
#distorted_points = distorted_points.reshape(height, width, 2)
#points = points.reshape(height, width, 2)
#tensor = 100 * (distorted_points - points)
#l_x, l_y, _ = tensor.shape
#
#subsample_factor = 10
#x, y = np.meshgrid(np.arange(0, l_x, subsample_factor), np.arange(0, l_y, subsample_factor))
#dx = tensor[::subsample_factor, ::subsample_factor, 0].numpy()
#dy = tensor[::subsample_factor, ::subsample_factor, 1].numpy()
#
## Visualization
#plt.figure(figsize=(25, 25))
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
