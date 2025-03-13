#!/usr/bin/env python3
import torch
import math



import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from dm_control import suite
from dm_control._render.executor import render_executor
from PIL import Image as PIL_Image
import pdb
import time



"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""


def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    """
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    """

    # This function is lifted directly from scipy source code
    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [
        x2 - y2 - z2 + w2,
        2 * (xy - zw),
        2 * (xz + yw),
        2 * (xy + zw),
        -x2 + y2 - z2 + w2,
        2 * (yz - xw),
        2 * (xz - yw),
        2 * (yz + xw),
        -x2 - y2 + z2 + w2,
    ]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat


"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""


def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat


"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""


def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat


"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


#
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""


class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """

    def __init__(self, env, min_bound=None, max_bound=None):
        super(PointCloudGenerator, self).__init__()

        self.env = env

        self.img_width = 64
        self.img_height = 64

        self.cam_names = [i for i in range(len(self.sim.model.cam_bodyid))]

        self.target_bounds = None
        if min_bound != None and max_bound != None:
            self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound, max_bound=max_bound
            )

        # List of camera intrinsic matrices
        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(
                ((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1))
            )
            self.cam_mats.append(cam_mat)

    def depthImageToPointCloud(self, depth_img, cam_id, max_depth = 6, down_sample_voxel_size=0.06) -> np.ndarray:

        od_cammat = cammat2o3d(
            self.cam_mats[cam_id], self.img_width, self.img_height
        )

        depth_img[depth_img >= max_depth] = 0

        od_depth = o3d.geometry.Image(np.ascontiguousarray(depth_img))

        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            od_depth, od_cammat
        )

        cam_pos = self.sim.model.cam_pos[cam_id]
        c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_id])
        b2w_r = quat2Mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        c2w = posRotMat2Mat(cam_pos, c2w_r)
        transformed_cloud = o3d_cloud.transform(c2w)

        if self.target_bounds != None:
            transformed_cloud = transformed_cloud.crop(self.target_bounds)

        points = np.asarray(transformed_cloud.points)

        transformed_cloud.points = o3d.utility.Vector3dVector(points)

        transformed_cloud =  transformed_cloud.voxel_down_sample(voxel_size=down_sample_voxel_size)

        points = np.asarray(transformed_cloud.points)

        np.random.shuffle(points) #so truncation isn't biased

        return points.astype(np.float32)

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image


    def save_point_cloud_as_image(self, point_cloud, output_image="point_cloud_image.png"):
        """Saves a 2D projection of the point cloud to an image using offscreen rendering."""
        # Offscreen rendering
        renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)

        # Add the point cloud to the scene
        renderer.scene.add_geometry("point_cloud", point_cloud, o3d.visualization.rendering.MaterialRecord())

        # Set the camera perspective
        renderer.scene.camera.look_at([0, 0, 0], [0, 0, 1], [0, 1, 0])

        # Render the scene and save as image
        image = renderer.render_to_image()
        o3d.io.write_image(output_image, image)


        # Clean up
        print(f"Point cloud projection saved to {output_image}")

    def save_point_cloud(self, point_cloud, is_point_cloud=True,output_file="point_cloud.ply"):

        if not is_point_cloud:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)

            point_cloud = pcd

        

        # Save the point cloud
        o3d.io.write_point_cloud("./point_cloud_images/" + output_file, point_cloud)
        print(f"Point cloud saved to {output_file}")


if __name__ == "__main__":
    pass
