import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from scene.dataset_readers import read_intrinsics_binary
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

def zero_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.)
        nn.init.constant_(m.bias, 0.)

def homogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (2, 3)
    return torch.cat(
        (X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)), dim=1
    )

def dehomogenize(X: torch.Tensor):
    assert X.ndim == 2
    assert X.shape[1] in (3, 4)
    return X[:, :-1] / X[:, -1:]

def colorize(uv_im, max_mag=None):
    hsv = np.zeros((uv_im.shape[0], uv_im.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(uv_im[..., 0], uv_im[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    if max_mag is None:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        mag = np.clip(mag, 0.0, max_mag)
        mag = mag / max_mag * 255.0
        hsv[..., 2] = mag.astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def plot_points(ref_points, path):
    p1 = ref_points.clone().reshape(-1, 2)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(int(ref_points.shape[1]/4), int(ref_points.shape[0]/4)))
    x = p1[:, 0].detach().cpu().numpy()  # Convert tensor to numpy for plotting
    y = p1[:, 1].detach().cpu().numpy()
    plt.scatter(x, y)
    plt.title('2D Points Plot')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.xlim(p1[:, 0].min().item() - 0.1, p1[:, 0].max().item() + 0.1)
    plt.ylim(p1[:, 1].min().item() - 0.1, p1[:, 1].max().item() + 0.1)
    plt.grid(True)
    plt.savefig(path)

def center_crop(tensor, target_height, target_width):
    _, _, height, width = tensor.size()

    # Calculate the starting coordinates for the crop
    start_y = (height - target_height) // 2
    start_x = (width - target_width) // 2

    # Create a grid for the interpolation
    grid_y, grid_x = torch.meshgrid(torch.linspace(start_y, start_y + target_height - 1, target_height),
                                    torch.linspace(start_x, start_x + target_width - 1, target_width))
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).to(tensor.device)

    # Normalize grid to [-1, 1]
    grid = 2.0 * grid / torch.tensor([width - 1, height - 1]).cuda() - 1.0
    grid = grid.permute(0, 1, 2, 3).expand(tensor.size(0), target_height, target_width, 2)

    # Perform the interpolation
    cropped_tensor = F.grid_sample(tensor, grid, align_corners=True)

    return cropped_tensor

def generate_pts(scene, boundary_scale=4, sample_resolution=20):
    viewpoint_cam = scene.getTrainCameras().copy()[0]
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    sample_width = int(width / sample_resolution)
    sample_height = int(height / sample_resolution)
    K = viewpoint_cam.get_K
    width = viewpoint_cam.fish_gt_image.shape[2]
    height = viewpoint_cam.fish_gt_image.shape[1]
    width = int(width * boundary_scale)
    height = int(height * boundary_scale)
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

    return P_sensor, P_view_insidelens_direction

def init_from_coeff(coeff, dataset, ref_points):
    r = torch.sqrt(torch.sum(ref_points**2, dim=-1, keepdim=True))
    inv_r = 1 / r
    theta = torch.atan(r)
    if os.path.exists(os.path.join(dataset.source_path, 'fish/sparse/0/cameras.bin')):
        cam_intrinsics = read_intrinsics_binary(os.path.join(dataset.source_path, 'fish/sparse/0/cameras.bin'))
        for idx, key in enumerate(cam_intrinsics):
            if 'RADIAL' in cam_intrinsics[key].model:
                coeff = cam_intrinsics[key].params[-2:].tolist()
            if 'FISHEYE' in cam_intrinsics[key].model:
                coeff = cam_intrinsics[key].params[-4:].tolist()
                break
    elif os.path.exists(os.path.join(dataset.source_path, 'cameras.json')):
        with open(os.path.join(dataset.source_path, 'cameras.json')) as json_file:
            contents = json.load(json_file)
            coeff = contents['KRT'][-1]['distortion']
    #coeff = [0., 0, 0, 0]
    if len(coeff) == 4:
        ref_points = ref_points * (inv_r * (theta + coeff[0] * theta**3 + coeff[1] * theta**5 + coeff[2] * theta**7 + coeff[3] * theta**9))
    elif len(coeff) == 2:
        ref_points = ref_points * (1 + coeff[0] * r**2 + coeff[1] * r**4)
    elif len(coeff) == 3:
        ref_points = ref_points * (1 + coeff[0] * r**2 + coeff[1] * r**4 + coeff[2] * r**6)
    elif len(coeff) == 8:
        x_n, y_n = ref_points[..., 0], ref_points[..., 1]
        p1, p2 = coeff[5], coeff[6]
        r_squared = x_n**2 + y_n**2
        tangential_distortion = torch.stack([
            2 * p1 * x_n * y_n + p2 * (r_squared + 2 * x_n**2),
            p1 * (r_squared + 2 * y_n**2) + 2 * p2 * x_n * y_n
        ], dim=-1)
        #ref_points = ref_points * (inv_r * (theta + coeff[0] * theta**3 + coeff[1] * theta**5 + coeff[2] * theta**7)) + tangential_distortion
        ref_points = ref_points * (inv_r * (theta + coeff[0] * theta**3 + coeff[1] * theta**5 + coeff[2] * theta**7))
    else:
        ref_points = ref_points

    return ref_points

def init_from_colmap(scene, dataset, optimizer_lens_net, lens_net, scheduler_lens_net, resume_training=None, iresnet_lr=1e-7):
    P_sensor, P_view_insidelens_direction = generate_pts(scene, boundary_scale=5, sample_resolution=40)
    P_view_outsidelens_direction = P_view_insidelens_direction
    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    ref_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
    coeff = [0, 0, 0, 0]
    ref_points = init_from_coeff(coeff, dataset, ref_points)
    inf_mask = torch.isinf(ref_points)
    nan_mask = torch.isnan(ref_points)
    ref_points[inf_mask] = 0
    ref_points[nan_mask] = 0
    plot_points(ref_points, os.path.join(scene.model_path, f"ref1.png"))

    P_sensor, P_view_insidelens_direction = generate_pts(scene, boundary_scale=1.5, sample_resolution=40)
    P_view_outsidelens_direction = P_view_insidelens_direction
    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    ref_points1 = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]
    coeff = [0, 0, 0, 0]
    ref_points1 = init_from_coeff(coeff, dataset, ref_points1)
    print(f"using coeff: {coeff}")
    inf_mask = torch.isinf(ref_points1)
    nan_mask = torch.isnan(ref_points1)
    ref_points1[inf_mask] = 0
    ref_points1[nan_mask] = 0
    plot_points(ref_points1, os.path.join(scene.model_path, f"ref2.png"))
    combine = torch.cat((ref_points, ref_points1), dim=0)
    plot_points(combine, os.path.join(scene.model_path, f"ref1_2.png"))

    P_sensor0, P_view_insidelens_direction0 = generate_pts(scene, boundary_scale=5, sample_resolution=40)
    P_sensor1, P_view_insidelens_direction1 = generate_pts(scene, boundary_scale=1.5, sample_resolution=40)
    P_sensor =torch.cat((P_sensor0, P_sensor1), dim=0)
    P_view_insidelens_direction =torch.cat((P_view_insidelens_direction0, P_view_insidelens_direction1), dim=0)

    if resume_training == None:
        progress_bar_ires = tqdm(range(0, 10000), desc="Init Iresnet")
        for i in range(10000):
            P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=True)
            control_points = homogenize(P_view_outsidelens_direction)
            inf_mask = torch.isinf(control_points)
            nan_mask = torch.isnan(control_points)
            control_points[inf_mask] = 0
            control_points[nan_mask] = 0
            loss = ((control_points[:, :2] - combine.reshape(-1, 2))**2).mean()
            progress_bar_ires.set_postfix(loss=loss.item())
            progress_bar_ires.update(1)
            loss.backward()
            optimizer_lens_net.step()
            optimizer_lens_net.zero_grad(set_to_none = True)
            scheduler_lens_net.step()

            if i % 2000 == 0:
                control_points_np = control_points.cpu().detach().numpy()
                ref_points_np = ref_points.reshape(-1, 2).cpu().detach().numpy()
                combine_np = combine.reshape(-1, 2).cpu().detach().numpy()
                plt.figure(figsize=(10, 6))
                plt.scatter(control_points_np[:, 0], control_points_np[:, 1], color='blue')
                plt.scatter(combine_np[:, 0], combine_np[:, 1], color='red')
                plt.savefig(f'/home/yd428/playaround_gaussian_platting/output/test/loss_{i}.png')
                plt.close()

        progress_bar_ires.close()

    for param_group in optimizer_lens_net.param_groups:
        param_group['lr'] = iresnet_lr
    print(f"The learning rate is reset to {param_group['lr']}")

def apply_distortion(lens_net, P_view_insidelens_direction, P_sensor, viewpoint_cam, image, apply2gt=False, flow_scale=None):
    P_view_outsidelens_direction = lens_net.forward(P_view_insidelens_direction, sensor_to_frustum=apply2gt)
    camera_directions_w_lens = homogenize(P_view_outsidelens_direction)
    control_points = camera_directions_w_lens.reshape((P_sensor.shape[0], P_sensor.shape[1], 3))[:, :, :2]

    if apply2gt:
        projection_matrix = viewpoint_cam.flow4gt
    else:
        projection_matrix = viewpoint_cam.projection_matrix
    flow = control_points @ projection_matrix[:2, :2]

    if apply2gt:
        flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(viewpoint_cam.image_height, viewpoint_cam.image_width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        gt_image = F.grid_sample(
            viewpoint_cam.fish_gt_image.cuda().unsqueeze(0),
            flow.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        mask = (~((gt_image[0]<0.00001) & (gt_image[1]<0.00001)).unsqueeze(0)).float()
        return gt_image, mask, flow
    else:
        flow = nn.functional.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(int(viewpoint_cam.fish_gt_image.shape[1]*flow_scale[0]), int(viewpoint_cam.fish_gt_image.shape[2]*flow_scale[1])), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        image = F.grid_sample(
            image.unsqueeze(0),
            flow.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        image = center_crop(image, viewpoint_cam.fish_gt_image.shape[1], viewpoint_cam.fish_gt_image.shape[2]).squeeze(0)
        mask = (~((image[0]==0.0000) & (image[1]==0.0000)).unsqueeze(0)).float()
        return image, mask, flow


def generate_control_pts(viewpoint_cam, control_point_sample_scale, flow_scale):
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height
    sample_width = int(width / control_point_sample_scale)
    sample_height = int(height/ control_point_sample_scale)
    K = viewpoint_cam.get_K
    width = viewpoint_cam.fish_gt_image.shape[2]
    height = viewpoint_cam.fish_gt_image.shape[1]
    width = int(width * flow_scale[0])
    height = int(height * flow_scale[1])
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

    return P_sensor, P_view_insidelens_direction
