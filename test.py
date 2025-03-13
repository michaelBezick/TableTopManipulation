import numpy as np
from PIL import Image
import robosuite as suite
from point_cloud_generator import PointCloudGenerator
import os

custom_xml_path = os.path.abspath("custom_xml_path")

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
    model=custom_xml_path,
)

breakpoint()

# pcg = PointCloudGenerator(env.physics)

# reset the environment
# breakpoint()

obs = env.reset()
for i in range(1):
    action = np.random.randn(env.robots[0].dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment

    breakpoint()


env.close()
