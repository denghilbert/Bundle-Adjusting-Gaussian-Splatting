import sys
import os
import numpy as np

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import torch

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):

    hwf = poses[:3, -1:, 0]

    center = poses[:3, 3, :].mean(-1)
    vec2 = normalize(poses[:3, 2, :].sum(-1))
    vec0_avg = poses[:3, 0, :].sum(-1)
    c2w = np.concatenate([viewmatrix(vec2, vec0_avg, center), hwf], 1)

    return c2w

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt


def nearest_pose(p, poses):
    dists = np.sum(np.square(p[:3, 3:4] - poses[:3, 3, :]), 0)
    return np.argsort(dists)



def render_path_axis(c2w, up, ax, rad, focal, N):
    render_poses = []
    center = c2w[:,3]
    hwf = c2w[:,4:5]
    v = c2w[:,ax] * rad
    for t in np.linspace(-1.,1.,N+1)[:-1]:
        c = center + t * v
        z = normalize(c - (center - focal * c2w[:,2]))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([-np.sin(theta), np.cos(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def generate_render_path(poses, focal=1, comps=None, N=30):
    if comps is None:
        comps = [True]*5
    #poses[:3, 1:3, :] *= -1

    # close_depth, inf_depth = bds[0, :].min()*.9, bds[1, :].max()*5.
    # dt = .75
    # mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    #focal = 30

    shrink_factor = .8
    zdelta = 0.8 #close_depth * .2

    c2w = poses_avg(poses)
    up = normalize(poses[:3, 0, :].sum(-1))

    tt = ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, -1)

    render_poses = []

    if comps[0]:
        render_poses += render_path_axis(c2w, up, 1, shrink_factor*rads[1], focal, N)
    if comps[1]:
        render_poses += render_path_axis(c2w, up, 0, shrink_factor*rads[0], focal, N)
    if comps[2]:
        render_poses += render_path_axis(c2w, up, 2, shrink_factor*zdelta, focal, N)

    rads[2] = zdelta
    if comps[3]:
        render_poses += render_path_spiral(c2w, up, rads, focal, zdelta, 0., 1, N*2)
        render_poses += render_path_spiral(c2w, up, rads, focal, zdelta, 0., 1, N*2)
        render_poses += render_path_spiral(c2w, up, rads, focal, zdelta, 0., 1, N*2)
    if comps[4]:
        render_poses += render_path_spiral(c2w, up, rads, focal, zdelta, .5, 2, N*4)

    render_poses = np.array(render_poses)

    #render_poses[:, :3, 1:3] *= -1
    return render_poses



def render_path_fig(poses, render_poses, scaling_factor=1., savepath=None):
    c2w = poses_avg(poses)
    # tt = pts2cam(poses, c2w)


    plt.figure(figsize=(12,4))

    plt.subplot(121)
    tt = ptstocam(render_poses[:,:3,3], c2w) * scaling_factor
    plt.plot(tt[:,1], -tt[:,0])
    tt = ptstocam(poses[:3,3,:].T, c2w) * scaling_factor
    plt.scatter(tt[:,1], -tt[:,0])
    plt.axis('equal')

    plt.subplot(122)
    tt = ptstocam(render_poses[:,:3,3], c2w) * scaling_factor
    plt.plot(tt[:,1], tt[:,2])
    tt = ptstocam(poses[:3,3,:].T, c2w) * scaling_factor
    plt.scatter(tt[:,1], tt[:,2])
    plt.axis('equal')

    if savepath is not None:
        plt.savefig(os.path.join(savepath, 'cam_path_slices.png'))
        plt.close()


def get_camera_wireframe(scale: float = 0.3):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    up1 = 0.5 * torch.tensor([0, 1.5, 4])
    up2 = 0.5 * torch.tensor([0, 2, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


def plot_cameras(ax, cameras, color: str = "blue"):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles


def plot_camera_scene(cameras, cameras_gt, status: str, out_path: str):
    """
    Plots a set of predicted cameras `cameras` and their corresponding
    ground truth locations `cameras_gt`. The plot is named with
    a string passed inside the `status` argument.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.clear()
    ax.set_title(status)
    handle_cam = plot_cameras(ax, cameras, color="#FF7D1E")
    handle_cam_gt = plot_cameras(ax, cameras_gt, color="#812CE5")
    plot_radius = 4
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([- plot_radius, plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    labels_handles = {
        "Novel cameras": handle_cam[0],
        "Input cameras": handle_cam_gt[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    # plt.show()
    plt.savefig(out_path)

    return fig
