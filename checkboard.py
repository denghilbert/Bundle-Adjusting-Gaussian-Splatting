from scene import Scene, GaussianModel, SpecularModel, iResNet
import torch
from torch import nn
import torch.nn.functional as F
from utils.util_distortion import homogenize, dehomogenize, colorize, plot_points, center_crop, init_from_colmap, apply_distortion, generate_control_pts
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

image = Image.open('/home/yd428/playaround_gaussian_platting/example_datasets/Checkerboard_pattern.svg.png')
image = Image.open('/home/yd428/playaround_gaussian_platting/example_datasets/OOAD630.jpg')
transform = transforms.ToTensor()
image = transform(image).cuda()

lens_net = iResNet().cuda()
lens_net = torch.load("/home/yd428/playaround_gaussian_platting/smerf/nyc_lr9_flow2_allimgs/lens_net30000.pth")
#lens_net = torch.load("/home/yd428/playaround_gaussian_platting/output/cube_lr7_allimgs/lens_net30000.pth")
#lens_net = torch.load("/home/yd428/playaround_gaussian_platting/eyeful/seating_lr7_optcam_allimgs/lens_net30000.pth")


width = 700
height = 700
sample_width = int(width / 2)
sample_height = int(height / 2)
K = torch.tensor([[3.8540e+02, 0.0000e+00, 1.3680e+03], [0.0000e+00, 3.8540e+02, 2.0480e+03], [0.0000e+00, 0.0000e+00, 1.0000e+00]], device='cuda:0')
K[0, 2] = width / 2
K[1, 2] = height / 2
i, j = np.meshgrid(
    np.linspace(0, width, sample_width),
    np.linspace(0, height, sample_height),
    indexing="ij",
)
i = i.T
j = j.T
P_sensor = (
    torch.from_numpy(np.stack((i, j), axis=-1))
    .to(torch.float32)
    .cuda()
)
P_sensor_hom = homogenize(P_sensor.reshape((-1, 2)))
P_view_insidelens_direction_hom = (torch.inverse(K) @ P_sensor_hom.T).T
P_view_insidelens_direction = dehomogenize(P_view_insidelens_direction_hom)
P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=False)
control_points = homogenize(P_view_outsidelens_direction)
control_points = control_points.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
inf_mask = torch.isinf(control_points)
nan_mask = torch.isnan(control_points)
control_points[inf_mask] = 0
control_points[nan_mask] = 0
projection_matrix = torch.tensor([[ 1.,  0.0000,  0.0000,  0.0000], [ 0.0000,  1.,  0.0000,  0.0000], [ 0.0000,  0.0000,  1.0001,  1.0000], [ 0.0000,  0.0000, -0.0100,  0.0000]], device='cuda:0')
flow = control_points @ projection_matrix[:2, :2]
#flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(2048, 2048), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(1164, 1164), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
image = F.grid_sample(
        image.unsqueeze(0),
        flow.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
        )
image = image.squeeze(0)
to_pil = transforms.ToPILImage()
pil_image = to_pil(image)
pil_image.save('/home/yd428/playaround_gaussian_platting/example_datasets/dis_check.png')
