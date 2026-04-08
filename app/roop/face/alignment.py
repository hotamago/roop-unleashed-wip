import cv2
import numpy as np
from skimage import transform as trans


WARP_TEMPLATE_SET = {
    "arcface_112_v1": np.array(
        [
            [0.35473214, 0.45658929],
            [0.64526786, 0.45658929],
            [0.50000000, 0.61154464],
            [0.37913393, 0.77687500],
            [0.62086607, 0.77687500],
        ],
        dtype=np.float32,
    ),
    "arcface_112_v2": np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089],
        ],
        dtype=np.float32,
    ),
    "arcface_128": np.array(
        [
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453],
        ],
        dtype=np.float32,
    ),
    "dfl_whole_face": np.array(
        [
            [0.35342266, 0.39285716],
            [0.62797622, 0.39285716],
            [0.48660713, 0.54017860],
            [0.38839287, 0.68750011],
            [0.59821427, 0.68750011],
        ],
        dtype=np.float32,
    ),
    "ffhq_512": np.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ],
        dtype=np.float32,
    ),
    "mtcnn_512": np.array(
        [
            [0.36562865, 0.46733799],
            [0.63305391, 0.46585885],
            [0.50019127, 0.61942959],
            [0.39032951, 0.77598822],
            [0.61178945, 0.77476328],
        ],
        dtype=np.float32,
    ),
    "styleganex_384": np.array(
        [
            [0.42353745, 0.52289879],
            [0.57725008, 0.52319972],
            [0.50123859, 0.61331904],
            [0.43364461, 0.68337652],
            [0.57015325, 0.68306005],
        ],
        dtype=np.float32,
    ),
}


arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _normalize_crop_size(image_size):
    if isinstance(image_size, (tuple, list, np.ndarray)):
        if len(image_size) != 2:
            raise ValueError("image_size must resolve to a width/height pair")
        return int(image_size[0]), int(image_size[1])
    size = int(image_size)
    return size, size


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    elif image_size % 128 == 0:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    elif image_size % 512 == 0:
        ratio = float(image_size) / 512.0
        diff_x = 32.0 * ratio
    else:
        ratio = float(image_size) / 112.0
        diff_x = 0

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]


def estimate_matrix_by_face_landmark_5(face_landmark_5, crop_size, warp_template="arcface_128"):
    crop_size = _normalize_crop_size(crop_size)
    template = WARP_TEMPLATE_SET.get(str(warp_template))
    landmark = np.asarray(face_landmark_5, dtype=np.float32)
    if landmark.shape != (5, 2):
        raise ValueError("face_landmark_5 must have shape (5, 2)")
    if template is None:
        if crop_size[0] != crop_size[1]:
            raise ValueError(f"Unknown warp template '{warp_template}' for non-square crop size {crop_size}")
        return estimate_norm(landmark, crop_size[0])
    template_points = template * np.asarray(crop_size, dtype=np.float32)
    matrix = cv2.estimateAffinePartial2D(
        landmark,
        template_points.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )[0]
    if matrix is None:
        if crop_size[0] != crop_size[1]:
            raise ValueError(f"Unable to estimate affine matrix for crop size {crop_size}")
        return estimate_norm(landmark, crop_size[0])
    return matrix.astype(np.float32)


def align_crop(img, landmark, image_size=112, mode="arcface"):
    crop_size = _normalize_crop_size(image_size)
    landmark = np.asarray(landmark, dtype=np.float32)
    if mode != "arcface":
        matrix = estimate_matrix_by_face_landmark_5(landmark, crop_size, mode)
        warped = cv2.warpAffine(
            img,
            matrix,
            crop_size,
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_AREA,
        )
        return warped, matrix
    if crop_size[0] != crop_size[1]:
        raise ValueError("arcface alignment requires a square crop size")
    matrix = estimate_norm(landmark, crop_size[0])
    warped = cv2.warpAffine(img, matrix, crop_size, borderValue=0.0)
    return warped, matrix


def square_crop(im, size):
    if im.shape[0] > im.shape[1]:
        height = size
        width = int(float(im.shape[1]) / im.shape[0] * size)
        scale = float(size) / im.shape[0]
    else:
        width = size
        height = int(float(im.shape[0]) / im.shape[1] * size)
        scale = float(size) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((size, size, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    matrix = t.params[0:2]
    cropped = cv2.warpAffine(data, matrix, (output_size, output_size), borderValue=0.0)
    return cropped, matrix


def trans_points2d(pts, matrix):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for index in range(pts.shape[0]):
        pt = pts[index]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(matrix, new_pt)
        new_pts[index] = new_pt[0:2]
    return new_pts


def trans_points3d(pts, matrix):
    scale = np.sqrt(matrix[0][0] * matrix[0][0] + matrix[0][1] * matrix[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for index in range(pts.shape[0]):
        pt = pts[index]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(matrix, new_pt)
        new_pts[index][0:2] = new_pt[0:2]
        new_pts[index][2] = pts[index][2] * scale
    return new_pts


def trans_points(pts, matrix):
    if pts.shape[1] == 2:
        return trans_points2d(pts, matrix)
    return trans_points3d(pts, matrix)


__all__ = [
    "align_crop",
    "estimate_matrix_by_face_landmark_5",
    "estimate_norm",
    "square_crop",
    "trans_points",
    "trans_points2d",
    "trans_points3d",
    "transform",
]
