from scipy.spatial.transform import Rotation as R

# Convert quaternion to Euler angles
quat = [0.56, 0.43, 0.43, 0.56]
euler_angles = R.from_quat(quat).as_euler('xyz', degrees=True)
print(euler_angles)  # Output: [roll, pitch, yaw]

