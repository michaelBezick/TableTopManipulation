from scipy.spatial.transform import Rotation as R

# Convert quaternion to Euler angles
quat = [0.56, 0.43, 0.43, 0.56]
euler_angles = R.from_quat(quat).as_euler('xyz', degrees=True)
print(euler_angles)  # Output: [roll, pitch, yaw]
euler_angles[1] -= 10  # Reduce pitch by 10 degrees
# Convert modified Euler angles back to quaternion
new_quat = R.from_euler('xyz', euler_angles, degrees=True).as_quat()
print("Updated Quaternion:", new_quat)
