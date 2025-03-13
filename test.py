import numpy as np
from PIL import Image
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift",  # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=1024,
    camera_widths=1024,
)

# reset the environment
obs = env.reset()
for i in range(1):
    action = np.random.randn(env.robots[0].dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    camera_image = obs["agentview_image"]
    image = Image.fromarray(camera_image)
    image.save("test.png")

env.close()
