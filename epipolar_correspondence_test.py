from projection_test import dist_point_point, dist_point_line, epipolar_correspondence_test
import torch
import cv2
import numpy as np


def mark_points(image, points, color, thickness):
    for x, y in points:
        cv2.circle(image, (round(x.item()), round(y.item())), 5, color, thickness)

if __name__ == "__main__":
    # epipolar_projection test
    ################# get some points #################
    # 0 xy [666, 402], [287, 156], [455, 786]
    # 1 xy [668, 224], [261, 182], [603, 670]
    points_img0 = torch.tensor([[666, 402],
                                [287, 156],
                                [455, 786]]).cuda()
    points_img1 = torch.tensor([[668, 224],
                                [261, 182],
                                [603, 670]]).cuda()
    # I project 3 pairs of correspondence points to 3d space
    points_correspondence = torch.tensor(
        [[[ 0.3300,  0.8574,  2.2788,  1.0000],
         [-0.8791,  1.4587,  1.3222,  1.0000],
         [ 1.1836,  2.2202,  2.2482,  1.0000]]]
    ).cuda()
    points_middle_correspondence_origin1 = torch.tensor(
        [[[ 0.3300,  0.8574,  2.2788,  1.0000],
         [-0.8791,  1.4587,  1.3222,  1.0000],
         [ 1.1836,  2.2202,  2.2482,  1.0000]]]
    ).cuda()
    ################# get some points #################
    ################# get extrinsic and intrinsic #################
    intrinsic0 = torch.tensor(
        [[1.1082e+03, 0.0000e+00, 3.9850e+02],
        [0.0000e+00, 1.1084e+03, 3.9900e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]]
    ).cuda()
    intrinsic1 = torch.tensor(
        [[1.1082e+03, 0.0000e+00, 3.9850e+02],
        [0.0000e+00, 1.1084e+03, 3.9900e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]]
    ).cuda()
    w2c0 = torch.tensor(
        [[0.1272, -0.5454,  0.8284, -0.5990],
        [ 0.8431,  0.4995,  0.1994, -1.1505],
        [-0.5225,  0.6731,  0.5234,  1.9745],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
    ).cuda()
    w2c1 = torch.tensor(
        [[0.4247, -0.4939,  0.7587, -0.4761],
        [ 0.3601,  0.8611,  0.3590, -2.3057],
        [-0.8307,  0.1207,  0.5435,  2.9239],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]
    ).cuda()
    ################# get extrinsic and intrinsic #################
    points_to_img0, points_to_img1, origin0_to_img1, origin1_to_img0 = epipolar_correspondence_test(points_correspondence, intrinsic0, intrinsic1, w2c0, w2c1)

    # find distance (quantitative)
    line_dists_0 = dist_point_line(points_img0, points_to_img0, origin1_to_img0)
    line_dists_1 = dist_point_line(points_img1, points_to_img1, origin0_to_img1)

    point_dists_0 = dist_point_point(points_img0, points_to_img0)
    point_dists_1 = dist_point_point(points_img1, points_to_img1)

    # visualize correspondence
    image0 = cv2.imread('epipolar_correspondence_img/0.png')
    mark_points(image0, points_img0, (255, 0, 0), -1) # solid circle
    mark_points(image0, points_to_img0, (0, 0, 255), 1) # hollow circle
    cv2.imwrite('epipolar_correspondence_img/correspondence_img0.png', image0)

    # visualize epipolar line
    image0 = cv2.imread('epipolar_correspondence_img/0.png')
    mark_points(image0, points_img0, (255, 0, 0), -1)
    points_to_img0 = np.around(points_to_img0.cpu().numpy()).astype(np.int32)
    origin1_to_img0 = np.around(origin1_to_img0.cpu().numpy()).astype(np.int32)
    points_to_img0 = 2 * points_to_img0 - origin1_to_img0
    for (start, end) in zip(points_to_img0, origin1_to_img0):
        start_point = tuple(start)
        end_point = tuple(end)
        image = cv2.line(image0, start_point, end_point, (0, 255, 0), 1)
    cv2.imwrite('epipolar_correspondence_img/epipolar_img0.png', image0)


