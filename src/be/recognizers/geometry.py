"""Inference-only geometric features used by FELF-SLR."""

from __future__ import annotations

import numpy as np


FACE_INDICES = [0, 2, 5, 9, 10]
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]
SPREAD_ANGLE_TRIPLETS = [
    (1, 0, 5),
    (5, 0, 9),
    (9, 0, 13),
    (13, 0, 17),
]
FINGER_CHAINS = [
    ((0, 1, 2, 3, 4), np.deg2rad([45.0, 80.0, 90.0]).astype(np.float32)),
    ((0, 5, 6, 7, 8), np.deg2rad([90.0, 130.0, 90.0]).astype(np.float32)),
    ((0, 9, 10, 11, 12), np.deg2rad([90.0, 130.0, 90.0]).astype(np.float32)),
    ((0, 13, 14, 15, 16), np.deg2rad([90.0, 130.0, 90.0]).astype(np.float32)),
    ((0, 17, 18, 19, 20), np.deg2rad([90.0, 130.0, 90.0]).astype(np.float32)),
]
SEQUENCE_LENGTH = 40


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _orthogonal_unit(vector: np.ndarray) -> np.ndarray:
    basis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(_unit(vector), basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    orthogonal = np.cross(vector, basis)
    if np.linalg.norm(orthogonal) <= 1e-8:
        orthogonal = np.cross(
            vector, np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
    return _unit(orthogonal)


def _rotate_about_axis(
    vector: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    axis = _unit(axis)
    cosine = np.float32(np.cos(angle))
    sine = np.float32(np.sin(angle))
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * np.dot(axis, vector) * (1.0 - cosine)
    ).astype(np.float32)


def _rotate_towards(
    base_direction: np.ndarray, target_direction: np.ndarray, bend: float
) -> np.ndarray:
    base_direction = _unit(base_direction)
    target_direction = _unit(target_direction)
    cross = np.cross(base_direction, target_direction)
    if np.linalg.norm(cross) <= 1e-8:
        if np.dot(base_direction, target_direction) >= 0:
            return base_direction
        cross = _orthogonal_unit(base_direction)
    return _unit(_rotate_about_axis(base_direction, cross, bend))


def rectify_hand_kinematics(
    hand_xyz: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    if hand_xyz.shape != (21, 3) or not np.any(hand_xyz):
        return hand_xyz.astype(np.float32)
    rectified = hand_xyz.copy().astype(np.float32)
    for chain, max_bends in FINGER_CHAINS:
        points = hand_xyz[list(chain)].astype(np.float32)
        if np.any(np.linalg.norm(points, axis=1) <= 1e-8):
            continue
        segment_lengths = np.linalg.norm(
            points[1:] - points[:-1], axis=1
        ).astype(np.float32)
        if np.any(segment_lengths <= 1e-8):
            continue
        new_points = points.copy()
        previous_direction = _unit(points[1] - points[0])
        for index in range(1, len(chain) - 1):
            raw_direction = _unit(points[index + 1] - points[index])
            bend = float(
                np.arccos(
                    np.clip(np.dot(previous_direction, raw_direction), -1.0, 1.0)
                )
            )
            bend = min(bend, float(max_bends[index - 1]))
            child_direction = _rotate_towards(
                previous_direction, raw_direction, bend
            )
            new_points[index + 1] = (
                new_points[index] + segment_lengths[index] * child_direction
            )
            previous_direction = _unit(
                new_points[index + 1] - new_points[index]
            )
        blended = points + alpha * (new_points - points)
        for local_index, joint_index in enumerate(chain[1:], start=1):
            rectified[joint_index] = blended[local_index]
    return rectified.astype(np.float32)


def _compute_angle(
    point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray
) -> np.float32:
    ba = point_a - point_b
    bc = point_c - point_b
    denominator = (np.linalg.norm(ba) + 1e-8) * (
        np.linalg.norm(bc) + 1e-8
    )
    cosine = np.dot(ba, bc) / denominator
    return np.float32(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compute_palm_normal(hand_xyz: np.ndarray, hand_label: str) -> np.ndarray:
    wrist, index_mcp, pinky_mcp = hand_xyz[0], hand_xyz[5], hand_xyz[17]
    if hand_label == "Right":
        first, second = index_mcp - wrist, pinky_mcp - wrist
    else:
        first, second = pinky_mcp - wrist, index_mcp - wrist
    return _unit(np.cross(first, second))


def pack_raw_frame(
    pose_landmarks: np.ndarray | None,
    left_hand_landmarks: np.ndarray | None,
    right_hand_landmarks: np.ndarray | None,
) -> np.ndarray:
    pose = (
        np.asarray(pose_landmarks, dtype=np.float32)
        if pose_landmarks is not None
        else np.zeros((33, 4), dtype=np.float32)
    )
    left = (
        np.asarray(left_hand_landmarks, dtype=np.float32)
        if left_hand_landmarks is not None
        else np.zeros((21, 3), dtype=np.float32)
    )
    right = (
        np.asarray(right_hand_landmarks, dtype=np.float32)
        if right_hand_landmarks is not None
        else np.zeros((21, 3), dtype=np.float32)
    )
    if pose.shape == (33, 3):
        pose = np.concatenate(
            [pose, np.ones((33, 1), dtype=np.float32)], axis=1
        )
    if pose.shape != (33, 4) or left.shape != (21, 3) or right.shape != (21, 3):
        raise ValueError(
            "Expected pose (33,4), left hand (21,3), and right hand (21,3)."
        )
    return np.concatenate(
        [pose.reshape(-1), left.reshape(-1), right.reshape(-1)]
    ).astype(np.float32)


def _prepare_parts(frame: np.ndarray, rectify_hands: bool = True):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (258,):
        raise ValueError(f"Expected a 258-D raw pose frame, got {frame.shape}.")
    pose = frame[:132].reshape(33, 4)[:, :3]
    left = frame[132:195].reshape(21, 3)
    right = frame[195:258].reshape(21, 3)
    if rectify_hands:
        left = rectify_hand_kinematics(left)
        right = rectify_hand_kinematics(right)
    return pose, left, right


def extract_part_aware_features(frame: np.ndarray):
    pose, left_hand, right_hand = _prepare_parts(frame)
    face = pose[FACE_INDICES]
    nose = face[0]
    face_scale = np.linalg.norm(face[1] - face[2])
    if face_scale <= 1e-6:
        face_scale = 1.0
    face_relative = ((face[1:] - nose) / face_scale).astype(np.float32)
    body_scale = np.linalg.norm(pose[11] - pose[12])
    if body_scale <= 1e-6:
        body_scale = 1.0

    def local_branch(hand_xyz: np.ndarray, hand_label: str) -> np.ndarray:
        valid = bool(np.any(hand_xyz != 0))
        relative = hand_xyz - hand_xyz[0]
        hand_scale = np.linalg.norm(hand_xyz[0] - hand_xyz[9])
        if not valid or hand_scale <= 1e-6:
            hand_scale = 1.0
        relative = relative / hand_scale
        bone_vectors = np.asarray(
            [relative[child] - relative[parent] for parent, child in HAND_BONES],
            dtype=np.float32,
        )
        bone_lengths = np.linalg.norm(bone_vectors, axis=1).astype(np.float32)
        flexion = np.asarray(
            [_compute_angle(relative[a], relative[b], relative[c])
             for a, b, c in FINGER_ANGLE_TRIPLETS],
            dtype=np.float32,
        )
        spread = np.asarray(
            [_compute_angle(relative[a], relative[b], relative[c])
             for a, b, c in SPREAD_ANGLE_TRIPLETS],
            dtype=np.float32,
        )
        palm = (
            compute_palm_normal(hand_xyz, hand_label)
            if valid
            else np.zeros(3, dtype=np.float32)
        )
        pinch = np.linalg.norm(hand_xyz[4] - hand_xyz[8]) / hand_scale
        fingertip_spread = np.mean(
            np.linalg.norm(
                hand_xyz[[8, 12, 16, 20]] - hand_xyz[[5, 9, 13, 17]],
                axis=1,
            )
        ) / hand_scale
        scale_ratio = hand_scale / body_scale
        return np.concatenate(
            [
                relative[1:].reshape(-1),
                bone_vectors.reshape(-1),
                bone_lengths,
                flexion,
                spread,
                palm,
                np.asarray(
                    [pinch, fingertip_spread, scale_ratio], dtype=np.float32
                ),
            ]
        ).astype(np.float32)

    left = local_branch(left_hand, "Left")
    right = local_branch(right_hand, "Right")
    wrist_global = np.concatenate([left_hand[0], right_hand[0]])
    inter_wrist = np.asarray(
        [np.linalg.norm(left_hand[0] - right_hand[0]) / body_scale],
        dtype=np.float32,
    )
    wrist_to_face = np.asarray(
        [
            np.linalg.norm(left_hand[0] - face[0]) / body_scale,
            np.linalg.norm(right_hand[0] - face[0]) / body_scale,
            np.linalg.norm(left_hand[0] - face[3]) / body_scale,
            np.linalg.norm(right_hand[0] - face[4]) / body_scale,
        ],
        dtype=np.float32,
    )
    global_features = np.concatenate(
        [face_relative.reshape(-1), wrist_global, inter_wrist, wrist_to_face]
    ).astype(np.float32)
    if left.shape != (165,) or right.shape != (165,) or global_features.shape != (23,):
        raise RuntimeError("Unexpected FELF-SLR feature dimensions.")
    return left, right, global_features


def extract_part_aware_sequence(sequence: np.ndarray):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape != (SEQUENCE_LENGTH, 258):
        raise ValueError(
            f"Expected raw sequence (40,258), got {sequence.shape}."
        )
    left, right, global_features = zip(
        *(extract_part_aware_features(frame) for frame in sequence)
    )
    return (
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
        np.asarray(global_features, dtype=np.float32),
    )


def _hand_morphology(hand: np.ndarray) -> np.ndarray:
    valid = float(np.any(np.abs(hand) > 1e-8))
    if valid <= 0:
        return np.zeros(155, dtype=np.float32)
    palm_center = hand[[0, 5, 9, 13, 17]].mean(axis=0)
    scale = float(np.linalg.norm(hand[0] - hand[9]))
    if scale <= 1e-6:
        scale = 1.0
    relative = ((hand - palm_center) / scale).astype(np.float32)
    bone_vectors = np.asarray(
        [relative[child] - relative[parent] for parent, child in HAND_BONES],
        dtype=np.float32,
    )
    bone_lengths = np.linalg.norm(bone_vectors, axis=1)
    fingertips = relative[[4, 8, 12, 16, 20]]
    spread = [
        np.linalg.norm(fingertips[i] - fingertips[j])
        for i in range(5)
        for j in range(i + 1, 5)
    ]
    return np.concatenate(
        [
            relative.reshape(-1),
            bone_vectors.reshape(-1),
            bone_lengths,
            np.asarray(spread, dtype=np.float32),
            np.asarray([valid, scale], dtype=np.float32),
        ]
    ).astype(np.float32)


def _hand_orientation(hand: np.ndarray, label: str) -> np.ndarray:
    valid = float(np.any(np.abs(hand) > 1e-8))
    if valid <= 0:
        return np.zeros(10, dtype=np.float32)
    return np.concatenate(
        [
            _unit(hand[5] - hand[17]),
            _unit(hand[9] - hand[0]),
            compute_palm_normal(hand, label),
            np.asarray([valid], dtype=np.float32),
        ]
    ).astype(np.float32)


def _frame_factor_features(frame: np.ndarray):
    pose, left, right = _prepare_parts(frame)
    face = pose[FACE_INDICES]
    face_center = face.mean(axis=0)
    face_scale = float(np.linalg.norm(face[1] - face[2]))
    shoulder_center = (pose[11] + pose[12]) * 0.5
    body_scale = float(np.linalg.norm(pose[11] - pose[12]))
    if face_scale <= 1e-6:
        face_scale = 1.0
    if body_scale <= 1e-6:
        body_scale = face_scale

    morphology = np.concatenate(
        [_hand_morphology(left), _hand_morphology(right)]
    ).astype(np.float32)
    orientation = np.concatenate(
        [_hand_orientation(left, "Left"), _hand_orientation(right, "Right")]
    ).astype(np.float32)

    def hand_trajectory(hand: np.ndarray, other: np.ndarray) -> np.ndarray:
        valid = float(np.any(np.abs(hand) > 1e-8))
        wrist = hand[0]
        palm = hand[[0, 5, 9, 13, 17]].mean(axis=0)
        return np.concatenate(
            [
                (wrist - shoulder_center) / body_scale,
                (palm - shoulder_center) / body_scale,
                (wrist - face_center) / face_scale,
                (palm - face_center) / face_scale,
                (wrist - other[0]) / body_scale,
                np.asarray(
                    [
                        np.linalg.norm(wrist - face_center) / body_scale,
                        np.linalg.norm(palm - face_center) / body_scale,
                        valid,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    left_trajectory = hand_trajectory(left, right)
    right_trajectory = hand_trajectory(right, left)
    inter = np.concatenate(
        [
            (left[0] - right[0]) / body_scale,
            (
                left[[0, 5, 9, 13, 17]].mean(axis=0)
                - right[[0, 5, 9, 13, 17]].mean(axis=0)
            )
            / body_scale,
            np.asarray(
                [np.linalg.norm(left[0] - right[0]) / body_scale],
                dtype=np.float32,
            ),
        ]
    )
    trajectory = np.concatenate(
        [left_trajectory, right_trajectory, inter]
    ).astype(np.float32)
    return morphology, trajectory, orientation


def extract_mt_sequence(sequence: np.ndarray, feature_version: str = "v1"):
    if feature_version != "v1":
        raise ValueError(
            "The ViSTAR runtime currently supports FELF_MT_FEATURE_VERSION=v1."
        )
    morphology_rows, trajectory_rows, orientation_rows = zip(
        *(_frame_factor_features(frame) for frame in sequence)
    )
    morphology = np.asarray(morphology_rows, dtype=np.float32)
    trajectory_base = np.asarray(trajectory_rows, dtype=np.float32)
    orientation_base = np.asarray(orientation_rows, dtype=np.float32)
    velocity = np.zeros_like(trajectory_base)
    velocity[1:] = trajectory_base[1:] - trajectory_base[:-1]
    acceleration = np.zeros_like(trajectory_base)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    orientation_delta = np.zeros_like(orientation_base)
    orientation_delta[1:] = orientation_base[1:] - orientation_base[:-1]
    curvature = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
    for frame_index in range(2, SEQUENCE_LENGTH):
        for hand_index, start in enumerate((0, 21)):
            previous = velocity[frame_index - 1, start:start + 3]
            current = velocity[frame_index, start:start + 3]
            denominator = float(np.linalg.norm(previous) * np.linalg.norm(current))
            if denominator > 1e-8:
                curvature[frame_index, hand_index] = (
                    1.0 - float(np.dot(previous, current) / denominator)
                )
    trajectory = np.concatenate(
        [trajectory_base, velocity, acceleration, curvature], axis=1
    ).astype(np.float32)
    orientation = np.concatenate(
        [orientation_base, orientation_delta], axis=1
    ).astype(np.float32)
    return morphology, trajectory, orientation
