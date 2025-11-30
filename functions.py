import numpy as np

MIN_X_DISTANCE = 5.0    # [cm] 로봇 앞에 물체가 너무 가까이 오지 않도록 최소 거리 설정
MAX_X_DISTANCE = 25.0   # [cm] 최대 수평 도달 거리
MAX_Y_RANGE = 15.0      # [cm] Px=20 근처에서 Y가 최대 20cm까지 움직이도록 제한
MIN_Z_HEIGHT = 10.0     # [cm] 최소 높이 (H=5cm)에 여유분 추가
MAX_Z_HEIGHT = 33.0     # [cm] 최대 높이 제한


def filter_del(del_x, del_y, del_z, prev_del_x, prev_del_y, prev_del_z):
    """
    delta x, y, z값 dead zone 설정 + 미세한 노이즈 필터링
    """
    if abs(del_x) < 15: del_x = 0
    if abs(del_y) < 20: del_y = 0
    if abs(del_z) < 5: del_z = 0


    if abs(del_x - prev_del_x) < 3:
        final_del_x = prev_del_x # 변화가 작으면 이전 값 유지
    else:
        final_del_x = del_x
        prev_del_x = final_del_x # 변화가 크면 새 값 승인 및 갱신

    if abs(del_y - prev_del_y) < 4:
        final_del_y = prev_del_y
    else:
        final_del_y = del_y
        prev_del_y = final_del_y

    if abs(del_z - prev_del_z) < 2:
        final_del_z = prev_del_z
    else:
        final_del_z = del_z
        prev_del_z = final_del_z
    
    return final_del_x, final_del_y, final_del_z


def limit_workspace(Px, Py, Pz, L2, L3, H):
    """
    IK 계산 전에 Px, Py, Pz 좌표가 로봇의 작업 공간 내에 있는지 확인하고 제한하는 함수.
    """
    # Px (거리) 제한
    if Px < MIN_X_DISTANCE:
        Px = MIN_X_DISTANCE
    elif Px > MAX_X_DISTANCE:
        Px = MAX_X_DISTANCE
    
    # Py (좌우 위치) 제한
    if Py > MAX_Y_RANGE:
        Py = MAX_Y_RANGE
    elif Py < -MAX_Y_RANGE:
        Py = -MAX_Y_RANGE

    # Pz (높이) 제한
    if Pz < MIN_Z_HEIGHT:
        Pz = MIN_Z_HEIGHT
    elif Pz > MAX_Z_HEIGHT:
        Pz = MAX_Z_HEIGHT
        
    # 도달 가능성 검증 (Pz, Px, Py의 조합 검증)
    r = np.sqrt(Px**2 + Py**2)
    dx = r
    dz = Pz - H
    
    total_dist = np.sqrt(dx**2 + dz**2)     # 끝점까지의 총 거리
    max_reach = L2 + L3                     # 팔을 완전히 폈을 때의 최대 거리 (L2 + L3)
    min_reach = 5.0                         # 팔을 완전히 접었을 때의 최소 거리 (안전상 5cm로 설정)

    if total_dist > max_reach:
        # 도달 불가능한 경우: Pz와 r을 비례적으로 줄여 최대 거리에 맞춤
        scale = max_reach / total_dist
        Px = Px * scale # Px, Py는 다시 계산해야 하지만, 간단하게 Px만 조정
        Py = Py * scale
        Pz = dz * scale + H # dz를 줄이고 H를 다시 더함
        
    elif total_dist < min_reach:
        # 너무 가까운 경우: Pz와 r을 비례적으로 늘려 최소 거리에 맞춤
        scale = min_reach / total_dist
        Px = Px * scale 
        Py = Py * scale
        Pz = dz * scale + H

    return Px, Py, Pz


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

    th1 = 90 + int(np.degrees(theta1))
    th2 = 180 - int(np.degrees(theta2))
    th3_ik = 180 + int(np.degrees(theta3))
    th3 = th2 + th3_ik -90

    return th1, th2, th3