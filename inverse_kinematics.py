import numpy as np

def inverse_kinematics(Px, Py, Pz, L2, L3, H):
    """
    3-DOF (Yaw-Pitch-Pitch) robot arm inverse kinematics.
    Returns theta1, theta2, theta3 in radians.
    """

    # --- Joint 1 (yaw) ---
    theta1 = np.arctan2(Py, Px)

    # --- Projection to x–z plane ---
    r = np.sqrt(Px**2 + Py**2)
    dx = r
    dz = Pz - H

    # --- IK for 2-link planar (theta2, theta3) ---
    D = (dx**2 + dz**2 - L2**2 - L3**2) / (2 * L2 * L3)
    # print("D:", D)
    
    if abs(D) > 1:
        raise ValueError("Target is outside reachable workspace.")

    # elbow-down solution (use -sqrt(...) )
    theta3 = np.arctan2(-np.sqrt(1 - D**2), D)

    theta2 = np.arctan2(dz, dx) - np.arctan2(L3*np.sin(theta3),
                                            L2 + L3*np.cos(theta3))

    theta1 = 90 + int(np.degrees(theta1))
    theta2 = 180 - int(np.degrees(theta2))
    theta3 = 180 + int(np.degrees(theta3))

    return theta1, theta2, theta3