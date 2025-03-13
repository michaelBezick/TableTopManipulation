import numpy as np
from PIL import Image
import robosuite as suite
from point_cloud_generator import PointCloudGenerator

# create environment instance
env = suite.make(
    env_name="Lift",  # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["custom_camera1", "custom_camera2"],
    camera_depths=True,
    camera_heights=128,
    camera_widths=128,
)

# Dynamically place cameras
env.sim.model.cam_pos[env.sim.model.camera_name2id("custom_camera1")] = [0.5, 0.0, 1.0]
env.sim.model.cam_pos[env.sim.model.camera_name2id("custom_camera2")] = [1.0, -0.5, 1.2]

env.sim.model.cam_quat[env.sim.model.camera_name2id("custom_camera1")] = [1, 0, 0, 0]  # Quaternion for orientation
env.sim.model.cam_quat[env.sim.model.camera_name2id("custom_camera2")] = [0.707, 0.707, 0, 0]  # 90-degree rotation

breakpoint()

# pcg = PointCloudGenerator(env.physics)

# reset the environment
# breakpoint()
obs = env.reset()
for i in range(1):
    action = np.random.randn(env.robots[0].dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment

    depth_image = obs["custom_camera1_image"]
    # pcd = pcg.depthImageToPointCloud(depth_image)

env.close()
