import cv2 as cv
import numpy as np
import open3d as o3d
from s2u.perception import camera_on_sphere
from s2u.utils.transform import Transform, Rotation
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.viz import draw_LAF_matches
import matplotlib.pyplot as plt



def loftr_matching(rgb_start_imgs, depth_start_imgs, rgb_target_imgs, depth_target_imgs, intrinsic):

    matcher = KF.LoFTR(pretrained="outdoor")

    size = 0.3
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, size / 2])
    n = 6
    N = n
    r = 1.2 * size

    extrinsics = []

    theta = np.pi / 8.0
    phi_list = 2.0 * np.pi * np.arange(n) / N
    extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

    theta = np.pi / 4.0
    phi_list = 2.0 * np.pi * np.arange(n) / N + 2.0 * np.pi / (N * 3)
    extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

    theta = np.pi / 2.0
    phi_list = 2.0 * np.pi * np.arange(n) / N + 4.0 * np.pi / (N * 3)
    extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

    # build Open3D intrinsics (must match your cv intrinsics)
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic.width,
        height=intrinsic.height,
        fx=intrinsic.fx,
        fy=intrinsic.fy,
        cx=intrinsic.cx,
        cy=intrinsic.cy,
    )

    all_correspondences = []  # list of (pts3d_start, pts3d_target)
    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # out = cv.VideoWriter("matches_output.mp4", fourcc, 20, (1280, 720))
    for i in range(len(rgb_start_imgs)-15):
        for j in range(len(rgb_target_imgs)-15):
            if j==i:
                continue
            else:
                print(f"i: {i}, j: {j}")
                # img1 = cv.cvtColor(rgb_start_imgs[i], cv.COLOR_BGR2GRAY)
                # img2 = cv.cvtColor(rgb_target_imgs[j], cv.COLOR_BGR2GRAY)
                img1 = rgb_start_imgs[i]
                img2 = rgb_target_imgs[j]
                img = np.hstack((img1, img2))
                # cv.imshow("images", img)
                # if cv.waitKey(0)==ord('s'):
                #     cv.destroyAllWindows
                H,W,C=img1.shape
                img1 = img1.reshape(C,H,W)[None, ...]
                img2 = img2.reshape(C,H,W)[None, ...]
                img1 = torch.from_numpy(img1).float() / 255.0
                img2 = torch.from_numpy(img2).float() / 255.0 
                # img1 = K.geometry.resize(img1, (H, W), antialias=True)
                # img2 = K.geometry.resize(img2, (H, W), antialias=True)               
                # img1 = K.image_to_tensor(img1, keepdim=False,)
                # img2 = K.image_to_tensor(img2, keepdim=False)


                input_dict = {
                            "image0": K.color.rgb_to_grayscale(img1),
                            "image1": K.color.rgb_to_grayscale(img2)}
                with torch.inference_mode():
                    correspondences = matcher(input_dict)
                
                pts1 = correspondences["keypoints0"].cpu().numpy()
                pts2 = correspondences["keypoints1"].cpu().numpy()
                if pts1.shape[0] >= 8 and pts2.shape[0] >= 8:  # need at least 8 points
                    Fm, inliers = cv.findFundamentalMat(
                        pts1, pts2, cv.USAC_MAGSAC, 1.0, 0.999, 100000
                    )
                    inliers = inliers > 0
                else:
                    Fm, inliers = None, None
                    print("Warning: Not enough correspondences found.")
                # Fm, inliers = cv.findFundamentalMat(pts1, pts2, cv.USAC_MAGSAC, 1.0, 0.999, 100000)
                
                

                draw_LAF_matches(
                    KF.laf_from_center_scale_ori(
                        torch.from_numpy(pts1).view(1, -1, 2),
                        torch.ones(pts1.shape[0]).view(1, -1, 1, 1),
                        torch.ones(pts1.shape[0]).view(1, -1, 1),
                    ),
                    KF.laf_from_center_scale_ori(
                        torch.from_numpy(pts2).view(1, -1, 2),
                        torch.ones(pts2.shape[0]).view(1, -1, 1, 1),
                        torch.ones(pts2.shape[0]).view(1, -1, 1),
                    ),
                    torch.arange(pts1.shape[0]).view(-1, 1).repeat(1, 2),
                    K.tensor_to_image(img1),
                    K.tensor_to_image(img2),
                    inliers,
                    draw_dict={
                        "inlier_color": (0.2, 1, 0.2),
                        "tentative_color": (1.0, 0.5, 1),
                        "feature_color": (0.2, 0.5, 1),
                        "vertical": False,
                    },
                )
                plt.show()



                
                # Prepare sparse color (3-channel uint8) and depth (float32 in meters)
                h, w = rgb_start_imgs[i].shape[:2]
                sparse_color_start = np.zeros((h, w, 3), dtype=np.uint8)
                sparse_color_target = np.zeros((h, w, 3), dtype=np.uint8)
                sparse_depth_start = np.ones((h, w), dtype=np.float32)
                sparse_depth_target = np.ones((h, w), dtype=np.float32)

                depth_start = depth_start_imgs[i]
                depth_target = depth_target_imgs[j]

                for (u1, v1), (u2, v2) in zip(pts1, pts2):
                    u1i, v1i = int(round(u1)), int(round(v1))
                    u2i, v2i = int(round(u2)), int(round(v2))

                    # Depth validity check (non-zero)
                    z1 = float(depth_start[v1i, u1i])
                    z2 = float(depth_target[v2i, u2i])

                    sparse_color_start[v1i, u1i, :] = rgb_start_imgs[i][v1i, u1i, :]
                    sparse_color_target[v2i, u2i, :] = rgb_target_imgs[j][v2i, u2i, :]


                    # ensure depth is in meters (float32)
                    sparse_depth_start[v1i, u1i] = z1
                    sparse_depth_target[v2i, u2i] = z2

                # Create Open3D RGBD images
                o3d_color_start = o3d.geometry.Image(sparse_color_start)
                o3d_color_target = o3d.geometry.Image(sparse_color_target)
                o3d_depth_start = o3d.geometry.Image(sparse_depth_start)  # float32 meters
                o3d_depth_target = o3d.geometry.Image(sparse_depth_target)

                rgbd_start = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color_start, o3d_depth_start, depth_scale=1.0, depth_trunc=6.0, convert_rgb_to_intensity=False
                )
                rgbd_target = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color_target, o3d_depth_target, depth_scale=1.0, depth_trunc=6.0, convert_rgb_to_intensity=False
                )

                # extrinsic: convert from your Transform to 4x4 numpy array (camera pose)
                extrinsic_start = extrinsics[i].as_matrix()  # should be 4x4, world <- camera or camera->world? see below
                extrinsic_target = extrinsics[j].as_matrix()

                # Create point clouds directly from RGBD (Open3D does the backprojection)
                pcd_start = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_start, o3d_intr, extrinsic_start)
                pcd_target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_target, o3d_intr, extrinsic_target)

                if len(pcd_start.points) != len(pcd_target.points):
                    if i==0:
                        correspondences_pc = np.ones(6)
                    continue

                pts3d_1 = np.asarray(pcd_start.points)
                pts3d_2 = np.asarray(pcd_target.points)

                if i==0 and j==1:
                    c = np.hstack((np.array(pts3d_1), np.array(pts3d_2)))
                    correspondences_pc = np.array(c)
                else:
                    c = np.hstack((np.array(pts3d_1), np.array(pts3d_2)))
                    print(c.shape)
                    correspondences_pc = np.vstack((correspondences_pc,c))

    return correspondences_pc



def orb_matching(rgb_start_imgs, depth_start_imgs, rgb_target_imgs, depth_target_imgs, intrinsic):
    # ORB
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    sift = cv.SIFT_create()

    size = 0.3
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, size / 2])
    n = 6
    N = n
    r = 1.2 * size

    extrinsics = []

    theta = np.pi / 8.0
    phi_list = 2.0 * np.pi * np.arange(n) / N
    extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

    theta = np.pi / 4.0
    phi_list = 2.0 * np.pi * np.arange(n) / N + 2.0 * np.pi / (N * 3)
    extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

    theta = np.pi / 2.0
    phi_list = 2.0 * np.pi * np.arange(n) / N + 4.0 * np.pi / (N * 3)
    extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

    # build Open3D intrinsics (must match your cv intrinsics)
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsic.width,
        height=intrinsic.height,
        fx=intrinsic.fx,
        fy=intrinsic.fy,
        cx=intrinsic.cx,
        cy=intrinsic.cy,
    )

    all_correspondences = []  # list of (pts3d_start, pts3d_target)
    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # out = cv.VideoWriter("matches_output.mp4", fourcc, 20, (1280, 720))
    for i in range(len(rgb_start_imgs)):
        for j in range(len(rgb_target_imgs)):
            if j==i:
                continue
            else:
                print(f"i: {i}, j: {j}")
                img1 = cv.cvtColor(rgb_start_imgs[i], cv.COLOR_BGR2RGB)
                img2 = cv.cvtColor(rgb_target_imgs[j], cv.COLOR_BGR2RGB)
                kp1, des1 = orb.detectAndCompute(img1, None)
                kp2, des2 = orb.detectAndCompute(img2, None)
                if des1 is None or des2 is None:
                    continue
                matches = bf.match(des1, des2)

                #matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

                # Optional: show matches
                final_img = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
                final_img = cv.resize(final_img, (1280, 720))
                cv.imshow("Matches", final_img)
                
                if cv.waitKey(1)==ord('s'):
                    cv.destroyAllWindows()


                if len(pts1) < 8:
                    continue

                F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 0.5, 0.99)
                if mask is None:
                    continue
                mask = mask.ravel().astype(bool)
                pts1 = pts1[mask]
                pts2 = pts2[mask]

                # Prepare sparse color (3-channel uint8) and depth (float32 in meters)
                h, w = rgb_start_imgs[i].shape[:2]
                sparse_color_start = np.zeros((h, w, 3), dtype=np.uint8)
                sparse_color_target = np.zeros((h, w, 3), dtype=np.uint8)
                sparse_depth_start = 10*np.zeros((h, w), dtype=np.float32)
                sparse_depth_target = 10*np.zeros((h, w), dtype=np.float32)

                depth_start = depth_start_imgs[i]
                depth_target = depth_target_imgs[j]

                for (u1, v1), (u2, v2) in zip(pts1, pts2):
                    u1i, v1i = int(round(u1)), int(round(v1))
                    u2i, v2i = int(round(u2)), int(round(v2))

                    # Depth validity check (non-zero)
                    z1 = float(depth_start[v1i, u1i])
                    z2 = float(depth_target[v2i, u2i])

                    sparse_color_start[v1i, u1i, :] = rgb_start_imgs[i][v1i, u1i, :]
                    sparse_color_target[v2i, u2i, :] = rgb_target_imgs[j][v2i, u2i, :]


                    # ensure depth is in meters (float32)
                    sparse_depth_start[v1i, u1i] = z1
                    sparse_depth_target[v2i, u2i] = z2

                # Create Open3D RGBD images
                o3d_color_start = o3d.geometry.Image(sparse_color_start)
                o3d_color_target = o3d.geometry.Image(sparse_color_target)
                o3d_depth_start = o3d.geometry.Image(sparse_depth_start)  # float32 meters
                o3d_depth_target = o3d.geometry.Image(sparse_depth_target)

                rgbd_start = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color_start, o3d_depth_start, depth_scale=1.0, depth_trunc=6.0, convert_rgb_to_intensity=False
                )
                rgbd_target = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color_target, o3d_depth_target, depth_scale=1.0, depth_trunc=6.0, convert_rgb_to_intensity=False
                )

                # extrinsic: convert from your Transform to 4x4 numpy array (camera pose)
                extrinsic_start = extrinsics[i].as_matrix()  # should be 4x4, world <- camera or camera->world? see below
                extrinsic_target = extrinsics[j].as_matrix()

                # Create point clouds directly from RGBD (Open3D does the backprojection)
                pcd_start = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_start, o3d_intr, extrinsic_start)
                pcd_target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_target, o3d_intr, extrinsic_target)

                if len(pcd_start.points) != len(pcd_target.points):
                    if i==0:
                        correspondences_pc = np.ones(6)
                    continue

                pts3d_1 = np.asarray(pcd_start.points)
                pts3d_2 = np.asarray(pcd_target.points)

                if i==0 and j==1:
                    c = np.hstack((np.array(pts3d_1), np.array(pts3d_2)))
                    correspondences_pc = np.array(c)
                else:
                    c = np.hstack((np.array(pts3d_1), np.array(pts3d_2)))
                    print(c.shape)
                    correspondences_pc = np.vstack((correspondences_pc,c))

    return correspondences_pc