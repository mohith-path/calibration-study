import numpy as np
from scipy.spatial.transform import Rotation as R


def to_transformation_matrix(poses: np.ndarray) -> np.ndarray:
    """
    Convert poses from [x, y, z, euler x, euler y, euler z] to a transformation matrix
    """
    transformations = []
    for pose in poses:
        pos = pose[:3]
        rot = pose[3:]
        rot = R.from_rotvec(rot, degrees=True).as_matrix()
        t = np.eye(4)
        t[:3, :3] = rot
        t[:3, 3] = pos
        transformations.append(t)

    return np.array(transformations)


def register(source: np.ndarray, target: np.ndarray, use_open3d: bool = False) -> np.ndarray:
    """
    Find the transformation target_T_source that takes points in the source frame to the target frame
    """
    if use_open3d:
        import o3d
        from open3d.pipelines.registration import registration_fgr_based_on_correspondence

        source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
        target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))

        corres = [[i, i] for i in range(40)]
        corres = o3d.utility.Vector2iVector(corres)

        source_T_target = registration_fgr_based_on_correspondence(source, target, corres).transformation

    else:
        # Take transpose as columns should be the points
        p1 = source.transpose()
        p2 = target.transpose()

        # Calculate centroids
        p1_mean = np.mean(p1, axis=1, keepdims=True)
        p2_mean = np.mean(p2, axis=1, keepdims=True)

        # Subtract centroids
        q1 = p1 - p1_mean
        q2 = p2 - p2_mean

        # Calculate covariance matrix
        H = np.matmul(q1, q2.transpose())

        # Calculate singular value decomposition (SVD)
        U, X, V_t = np.linalg.svd(H)  # the SVD of linalg gives you Vt

        # Calculate rotation matrix
        R = np.matmul(V_t.transpose(), U.transpose())

        assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

        # Calculate translation matrix
        translation = p2_mean - np.matmul(R, p1_mean)

        source_T_target = np.eye(4)
        source_T_target[:3, :3] = R
        source_T_target[:3, 3:] = translation

    return source_T_target


def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points (3 x 1) into homogenous points (4 x 1)
    """
    shape = points.shape[:-1]
    points = np.concatenate((points, np.ones((*shape, 1), dtype=points.dtype)), axis=-1)
    return points


def compute_errors(source: np.ndarray, target: np.ndarray, target_T_source: np.ndarray) -> np.ndarray:
    """
    Compute residual error (target_points - target_T_source * source_points)
    """
    transformed_pts = np.matmul(target_T_source, homogenize(source).T)
    error = transformed_pts.T[:, :-1] - target
    error = np.linalg.norm(error, axis=-1)
    return error
