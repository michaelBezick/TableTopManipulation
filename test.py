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
    camera_depths=True,
    camera_heights=128,
    camera_widths=128,
)

# reset the environment
# breakpoint()
obs = env.reset()
for i in range(1):
    action = np.random.randn(env.robots[0].dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment

    depth_image = obs["agentview_depth"]
    # Normalize depth to 0-255 for saving
    depth_image = (depth_image / depth_image.max() * 255).astype(np.uint8)

    depth_image = np.squeeze(depth_image)

    # Save depth image
    depth_pil = Image.fromarray(depth_image)
    depth_pil.save(f"depth_frame_{i}.png")

env.close()
