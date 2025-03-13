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
    camera_names=['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'],
    camera_depths=True,
    camera_heights=128,
    camera_widths=128,
)

breakpoint()

# pcg = PointCloudGenerator(env.physics)

# reset the environment
# breakpoint()

camera_names = ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand']

for camera_name in camera_names:
    obs = env.reset()
    for i in range(1):
        action = np.random.randn(env.robots[0].dof)  # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment

        breakpoint()


env.close()
