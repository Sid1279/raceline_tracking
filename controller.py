import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack


def estimate_upcoming_curvature(centerline, start_idx, sample_distance=30, window=2):
    """Sample curvature over a distance ahead to anticipate turns"""
    N = len(centerline)
    curvatures = []
    dist_acc = 0.0
    idx = start_idx

    while dist_acc < sample_distance and len(curvatures) < 15:
        idx_prev = (idx - window) % N
        idx_next = (idx + window) % N

        p_prev = centerline[idx_prev]
        p_curr = centerline[idx % N]
        p_next = centerline[idx_next]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        cross_prod = np.abs(np.cross(v1, v2))
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        curv = cross_prod / (norms + 1e-9)
        curvatures.append(curv)

        # Advance
        p1 = centerline[idx % N]
        p2 = centerline[(idx + 1) % N]
        dist_acc += np.linalg.norm(p2 - p1)
        idx += 1

    # Return weighted average (prioritize near-term curvature)
    if not curvatures:
        return 0.0
    weights = np.linspace(1.0, 0.4, len(curvatures))
    return np.average(curvatures, weights=weights)


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """Enhanced Pure Pursuit with adaptive lookahead and predictive speed planning"""

    # Unpack state
    x, y = state[0:2]
    current_steering = state[2]
    current_velocity = state[3]
    heading = state[4]

    wheelbase = parameters[0]
    max_steering = parameters[4]

    centerline = racetrack.centerline
    N = len(centerline)

    # Find closest point on track
    distances = np.linalg.norm(centerline - np.array([x, y]), axis=1)
    closest_idx = np.argmin(distances)
    lateral_error = np.min(distances)

    # Adaptive lookahead: varies with speed but stays reasonable
    # Higher speed -> look further ahead (better stability)
    speed_factor = np.clip(current_velocity / 40.0, 0.3, 1.0)
    lookahead_dist = 12.0 + speed_factor * 15.0  # Range: 12-27 meters

    dist_acc = 0.0
    idx = closest_idx

    while dist_acc < lookahead_dist and idx < closest_idx + N:
        p1 = centerline[idx % N]
        p2 = centerline[(idx + 1) % N]
        dist_acc += np.linalg.norm(p2 - p1)
        idx += 1

    target_point = centerline[idx % N]

    # Calculate desired heading to target
    dx = target_point[0] - x
    dy = target_point[1] - y
    desired_heading = np.arctan2(dy, dx)

    # Angle to target (alpha in pure pursuit)
    alpha = desired_heading - heading
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    # Pure Pursuit steering formula
    desired_steering = np.arctan2(2 * wheelbase * np.sin(alpha), lookahead_dist)

    # Gentle centerline correction when drifting too far
    if lateral_error > 2.5:
        # Path direction at closest point
        next_idx = (closest_idx + 1) % N
        path_vec = centerline[next_idx] - centerline[closest_idx]
        path_vec = path_vec / (np.linalg.norm(path_vec) + 1e-9)

        to_car = np.array([x, y]) - centerline[closest_idx]
        cross_track_error = np.cross(path_vec, to_car)

        # Small correction proportional to drift
        correction = -0.08 * np.tanh(cross_track_error / 3.0)
        desired_steering += correction

    desired_steering = np.clip(desired_steering, -max_steering, max_steering)

    # Predictive speed planning: look ahead at upcoming curvature
    upcoming_curvature = estimate_upcoming_curvature(
        centerline, closest_idx, sample_distance=45
    )

    # Speed model with corner anticipation
    v_max = 47.5  # Max speed on straights
    k_base = 10.0

    # Increase k_base if sharp turn is coming (brake earlier)
    if upcoming_curvature > 0.15:
        k_base += 8.0
    elif upcoming_curvature > 0.08:
        k_base += 4.0

    target_velocity = v_max / (1.0 + k_base * upcoming_curvature)
    target_velocity = np.clip(target_velocity, 14.0, v_max)

    # Reduce speed based on current steering angle
    steering_factor = abs(desired_steering) / max_steering
    target_velocity *= 1.0 - 0.25 * steering_factor

    # Smooth velocity changes: limit how quickly target can change
    # This creates more predictable acceleration/braking
    max_velocity_change = 8.0  # m/s per control step
    if hasattr(controller, "prev_target_velocity"):
        velocity_diff = target_velocity - controller.prev_target_velocity
        velocity_diff = np.clip(
            velocity_diff, -max_velocity_change, max_velocity_change
        )
        target_velocity = controller.prev_target_velocity + velocity_diff

    controller.prev_target_velocity = target_velocity

    return np.array([desired_steering, target_velocity])


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """Low-level controller with smooth control outputs"""

    current_steering = state[2]
    current_velocity = state[3]

    desired_steering = desired[0]
    desired_velocity = desired[1]

    # Proportional steering rate control with slight damping
    Kp_steering = 5.0
    steering_error = desired_steering - current_steering
    steering_rate = Kp_steering * steering_error

    # Add slight damping to reduce oscillation
    if hasattr(lower_controller, "prev_steering_error"):
        steering_derivative = (
            steering_error - lower_controller.prev_steering_error
        ) / 0.1
        steering_rate -= 0.15 * steering_derivative

    lower_controller.prev_steering_error = steering_error
    steering_rate = np.clip(steering_rate, parameters[7], parameters[9])

    # Proportional velocity control
    Kp_velocity = 1.4
    velocity_error = desired_velocity - current_velocity
    acceleration = Kp_velocity * velocity_error
    acceleration = np.clip(acceleration, parameters[8], parameters[10])

    return np.array([steering_rate, acceleration])