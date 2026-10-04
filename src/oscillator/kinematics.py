"""K1 foot diagnostics using the viewer's pinned hierarchy, without pose alignment."""

import numpy as np
from scipy.spatial.transform import Rotation


def feet_world(skeleton, joints, root_pos, root_quat):
    positions, rotations = [], []
    for body in skeleton["bodies"]:
        parent = body["parent"]
        if parent < 0:
            positions.append(root_pos)
            rotations.append(Rotation.from_quat(root_quat))
            continue
        quat = np.asarray(body["quaternion_wxyz"])[[1, 2, 3, 0]]
        rotation = Rotation.from_quat(quat)
        position = np.broadcast_to(np.asarray(body["position"], dtype=float), root_pos.shape).copy()
        if body["joint"] is not None:
            hinge = Rotation.from_rotvec(joints[:, body["joint"], None] * body["axis"])
            pivot = np.asarray(body["pivot"])
            position += rotation.apply(pivot - hinge.apply(pivot))
            rotation = rotation * hinge
        positions.append(positions[parent] + rotations[parent].apply(position))
        rotations.append(rotations[parent] * rotation)
    return np.stack([positions[index] for index in skeleton["feet"]], axis=1)


def foot_metrics(skeleton, prediction, root_pos, root_quat, clip, mask):
    feet = feet_world(skeleton, prediction[:, :22], root_pos, root_quat)
    source = clip["feet_world"][mask]
    consecutive = np.diff(np.flatnonzero(mask)) == 1
    contacts = (clip["contacts"][mask][1:] > 0.5) & (clip["contacts"][mask][:-1] > 0.5)
    contacts &= consecutive[:, None]
    result = {
        "foot_position_rmse_m": float(np.sqrt(np.mean(np.sum((feet - source) ** 2, -1)))),
        "contact_samples": int(contacts.sum()),
    }
    for key, positions in (("learned", feet), ("source", source)):
        speed = np.linalg.norm(np.diff(positions[..., :2], axis=0), axis=-1) * float(clip["fps"])
        result[f"{key}_contact_horizontal_speed_rms_m_s"] = (
            float(np.sqrt(np.mean(speed[contacts] ** 2))) if contacts.any() else None
        )
    return result
