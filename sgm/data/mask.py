# Copyright (c) OpenMMLab. All rights reserved.

"""
Functions taken from https://github.com/DanBigioi/DiffusionVideoEditing


"""

import cv2
import numpy as np
import torch

" Countour from 2:15 not good for head poses "


def face_mask(img_shape, landmark_list, dtype="uint8"):
    height, width = img_shape[:2]
    mask = np.ones((height, width, 1), dtype=dtype)
    cv2.drawContours(
        mask, np.int32([landmark_list[2:15]]), -1, color=(0), thickness=cv2.FILLED
    )

    return mask


def face_mask_jaw_box(img_shape, landmark_list, dtype="uint8", kernel_size=10):
    nose = 33
    jaw = 8

    height, width = img_shape[:2]
    mask = np.ones((height, width, 1), dtype=dtype)
    combined_landmarks = np.concatenate((landmark_list[2:15], [landmark_list[33]]))

    # Draw the combined contour on the mask
    cv2.drawContours(
        mask, [np.int32(combined_landmarks)], -1, color=(0), thickness=cv2.FILLED
    )

    inverted_mask = 1 - mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(inverted_mask, kernel, iterations=1)
    mask = np.expand_dims(
        mask, axis=-1
    )  # Add a singleton dimension to match the number of channels
    mask = 1 - mask

    cut_h = landmark_list[nose][1]

    far_left = int(np.argmin(landmark_list[:, 0]))
    far_right = int(np.argmax(landmark_list[:, 0]))
    left_up_point = np.int32([landmark_list[far_left][0], cut_h])  # 2
    right_up_point = np.int32([landmark_list[far_right][0], cut_h])  # 15
    height_landmarks = min(landmark_list[jaw, 1] + 20, height)
    left_down_point = np.int32([landmark_list[far_left][0], height_landmarks])
    right_down_point = np.int32([landmark_list[far_right][0], height_landmarks])

    # print(cut_h, cut_h + 10, height_landmarks)

    mask_box = [left_up_point, left_down_point, right_down_point, right_up_point]

    return mask, mask_box


" Stretch the tight face mask - Countour from 2:15 but dilate, not good for extreme head poses "


def face_mask_stretch(img_shape, landmark_list, dtype="uint8", kernel_size=10):
    height, width = img_shape[:2]
    mask = np.ones((height, width, 1), dtype=dtype)
    combined_landmarks = np.concatenate((landmark_list[2:15], [landmark_list[33]]))

    # Draw the combined contour on the mask
    cv2.drawContours(
        mask, [np.int32(combined_landmarks)], -1, color=(0), thickness=cv2.FILLED
    )

    # cv2.drawContours(mask, np.int32([landmark_list[2:15]]), -1, color=(0), thickness=cv2.FILLED)
    inverted_mask = 1 - mask

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(inverted_mask, kernel, iterations=1)
    mask = np.expand_dims(
        mask, axis=-1
    )  # Add a singleton dimension to match the number of channels
    mask = 1 - mask

    return mask


" Small box around mouth - Use far left, far right points for extreme head poses, cut between nose and upper mouth point"


def face_mask_box_pose(img_shape, landmark_list, dtype="uint8"):
    """
    When the head pose is different than frontal then the normal cropping with landmarks does not work correctly.
    Crop using as height the middle nose point
    Take the left/right corners using the far_left and far_right landmarks
    TODO: Maybe it is better to add some more pixels to have a bigger mask, especially on large head poses
    """

    height, width = img_shape[:2]

    nose = 33
    upper_lip = 51
    jaw = 8

    nose_point_h = landmark_list[nose, 1]
    upper_lip_point = landmark_list[upper_lip, 1]
    cut_h = (upper_lip_point - nose_point_h) / 2 + nose_point_h

    # cut_h = landmark_list[nose][1]

    mask = np.ones((height, width, 1), dtype=dtype)

    far_left = int(np.argmin(landmark_list[:, 0]))
    far_right = int(np.argmax(landmark_list[:, 0]))

    left_up_point = np.int32([landmark_list[far_left][0], cut_h])  # 2
    right_up_point = np.int32([landmark_list[far_right][0], cut_h])  # 15

    height_landmarks = min(landmark_list[jaw, 1] + 20, height)
    left_down_point = np.int32([landmark_list[far_left][0], height_landmarks])
    right_down_point = np.int32([landmark_list[far_right][0], height_landmarks])

    cv2.drawContours(
        mask,
        np.int32(
            [
                [
                    left_up_point,
                    left_down_point,
                    right_up_point,
                    right_down_point,
                    left_up_point,
                    right_up_point,
                    left_down_point,
                    right_down_point,
                ]
            ]
        ),
        -1,
        color=(0),
        thickness=cv2.FILLED,
    )

    return mask


" Small box around mouth - Use far left, far right points for extreme head poses, cut from nose"


def face_mask_box_pose_nose(
    img_shape,
    landmark_list,
    dtype="uint8",
    get_box=False,
    pixels_above_nose=None,
    pixels_under_jaw=None,
):
    height, width = img_shape[:2]

    nose = 33
    jaw = 8

    cut_h = landmark_list[nose][1]
    if pixels_above_nose is not None:
        # this is only for inference to take a bigger mask and blend it back to the original frame
        cut_h = cut_h - pixels_above_nose

    mask = np.ones((height, width, 1), dtype=dtype)

    far_left = int(np.argmin(landmark_list[:, 0]))
    far_right = int(np.argmax(landmark_list[:, 0]))

    left_up_point = np.int32([landmark_list[far_left][0], cut_h])  # 2
    right_up_point = np.int32([landmark_list[far_right][0], cut_h])  # 15

    height_landmarks = min(landmark_list[jaw, 1] + 20, height)
    if pixels_under_jaw is not None:
        height_landmarks = min(landmark_list[jaw, 1] + pixels_under_jaw, height)
    left_down_point = np.int32([landmark_list[far_left][0], height_landmarks])
    right_down_point = np.int32([landmark_list[far_right][0], height_landmarks])

    cv2.drawContours(
        mask,
        np.int32(
            [
                [
                    left_up_point,
                    left_down_point,
                    right_up_point,
                    right_down_point,
                    left_up_point,
                    right_up_point,
                    left_down_point,
                    right_down_point,
                ]
            ]
        ),
        -1,
        color=(0),
        thickness=cv2.FILLED,
    )

    if get_box:
        mask_box = [left_up_point, left_down_point, right_down_point, right_up_point]
        return mask, mask_box
    else:
        return mask


def face_mask_box_pose_big(
    img_shape, landmark_list, dtype="uint8", cut_h=None, far_left=None, far_right=None
):
    height, width = img_shape[:2]
    mask = np.ones((height, width, 1), dtype=dtype)
    nose = 33
    nose_point_h = landmark_list[nose, 1]
    if cut_h is None:
        cut_h = nose_point_h

    if far_right is None and far_left is None:
        far_left = int(np.argmin(landmark_list[:, 0]))
        far_right = int(np.argmax(landmark_list[:, 0]))

        left_up_point = np.int32([landmark_list[far_left][0], cut_h])
        left_down_point = np.int32([landmark_list[far_left][0], height])

        right_up_point = np.int32([landmark_list[far_right][0], cut_h])
        right_down_point = np.int32([landmark_list[far_right][0], height])
    else:
        left_up_point = np.int32([far_left, cut_h])
        left_down_point = np.int32([far_left, height])

        right_up_point = np.int32([far_right, cut_h])
        right_down_point = np.int32([far_right, height])

    cv2.drawContours(
        mask,
        np.int32(
            [
                [
                    left_up_point,
                    left_down_point,
                    right_up_point,
                    right_down_point,
                    left_up_point,
                    right_up_point,
                    left_down_point,
                    right_down_point,
                ]
            ]
        ),
        -1,
        color=(0),
        thickness=cv2.FILLED,
    )

    return mask


def face_mask_box_pose_big_cover_nose(img_shape, landmark_list, dtype="uint8"):
    height, width = img_shape[:2]

    middle_nose_point = 29

    cut_h = landmark_list[middle_nose_point, 1]

    mask = np.ones((height, width, 1), dtype=dtype)

    far_left = int(np.argmin(landmark_list[:, 0]))
    far_right = int(np.argmax(landmark_list[:, 0]))

    left_up_point = np.int32([landmark_list[far_left][0], cut_h])
    left_down_point = np.int32([landmark_list[far_left][0], height])

    right_up_point = np.int32([landmark_list[far_right][0], cut_h])
    right_down_point = np.int32([landmark_list[far_right][0], height])

    cv2.drawContours(
        mask,
        np.int32(
            [
                [
                    left_up_point,
                    left_down_point,
                    right_up_point,
                    right_down_point,
                    left_up_point,
                    right_up_point,
                    left_down_point,
                    right_down_point,
                ]
            ]
        ),
        -1,
        color=(0),
        thickness=cv2.FILLED,
    )

    return mask


def face_mask_square(img_shape, landmark_list, dtype="uint8"):
    height, width = img_shape[:2]

    mask = np.ones((height, width, 1), dtype=dtype)

    far_left = np.min(landmark_list[:, 0])
    far_right = np.max(landmark_list[:, 1])
    print("far_left {}, far_right {}".format(far_left, far_right))

    left_p = 2
    right_p = 14

    print(
        "left_p {}, right_p {}".format(
            landmark_list[left_p][0], landmark_list[right_p][0]
        )
    )

    cv2.drawContours(
        mask,
        np.int32(
            [
                [
                    landmark_list[left_p],
                    [landmark_list[left_p][0], height],
                    landmark_list[right_p],
                    [landmark_list[right_p][0], height],
                    landmark_list[left_p],
                    landmark_list[right_p],
                    [landmark_list[left_p][0], height],
                    [landmark_list[right_p][0], height],
                ]
            ]
        ),
        -1,
        color=(0),
        thickness=cv2.FILLED,
    )

    return mask


" Used for half face "


def bbox2mask(img_shape, bbox, dtype="uint8"):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
            img_shape (tuple[int]): The size of the image.
            bbox (tuple[int]): Configuration tuple, (top, left, height, width)
            dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
            numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.ones((height, width, 1), dtype=dtype)
    mask[bbox[0] : bbox[0] + bbox[2], bbox[1] : bbox[1] + bbox[3], :] = 0.0

    return mask


def face_mask_cheeks(img_shape, landmark_list, dtype="uint8"):
    height, width = img_shape[:2]
    mask = np.ones((height, width, 1), dtype=dtype)

    middle_nose_point = 29
    nose = 33
    cut_h = int(landmark_list[middle_nose_point, 1])

    far_left = int(np.argmin(landmark_list[:, 0]))
    far_right = int(np.argmax(landmark_list[:, 0]))

    left_up_point = np.int32([landmark_list[far_left][0], cut_h])
    left_down_point = np.int32([landmark_list[far_left][0], height])

    right_up_point = np.int32([landmark_list[far_right][0], cut_h])
    right_down_point = np.int32([landmark_list[far_right][0], height])

    cv2.drawContours(
        mask,
        np.int32(
            [
                [
                    left_up_point,
                    left_down_point,
                    right_up_point,
                    right_down_point,
                    left_up_point,
                    right_up_point,
                    left_down_point,
                    right_down_point,
                ]
            ]
        ),
        -1,
        color=(0),
        thickness=cv2.FILLED,
    )

    #  Calculate the bounding box coordinates for the nose
    nose_jaw_dist = (
        abs(landmark_list[2][0] - landmark_list[middle_nose_point][0]) * 0.10
    )  # 1, 15
    # nose_right_dist = (landmark_list[middle_nose_point][0] - landmark_list[1][0]) * 0.10
    # nose_left_dist = (landmark_list[15][0] - landmark_list[middle_nose_point][0]) * 0.10
    #

    nose_min_x = int(landmark_list[31][0] - nose_jaw_dist)
    nose_max_x = int(landmark_list[35][0] + nose_jaw_dist)
    # nose_min_x = int(landmark_list[31][0] - nose_right_dist)
    # nose_max_x = int(landmark_list[35][0] + nose_left_dist)
    nose_min_y = cut_h
    nose_max_y = int(landmark_list[nose, 1])

    # Clear the nose area from the mask using a rectangle
    mask_nose = np.ones((height, width, 1), dtype=dtype)
    cv2.rectangle(
        mask_nose,
        (nose_min_x, nose_min_y),
        (nose_max_x, nose_max_y),
        color=(0),
        thickness=cv2.FILLED,
    )

    mask_nose = 1 - mask_nose
    mask = mask + mask_nose

    return mask


def face_mask_cheeks_batch(
    img_shape, landmark_list, dtype="uint8", box_expand=0.0, show_nose=True
):
    height, width = img_shape[:2]

    # Handle both single and multiple landmarks
    if len(landmark_list.shape) == 2:
        landmark_list = landmark_list[None, ...]  # Add batch dimension
    num_frames = landmark_list.shape[0]

    # Initialize masks for all frames
    masks = np.ones((num_frames, height, width), dtype=dtype)

    for i in range(num_frames):
        landmarks = landmark_list[i]
        middle_nose_point = 29
        nose = 33
        cut_h = int(landmarks[middle_nose_point, 1])

        # Add height expansion
        if box_expand > 0:
            cut_h = max(0, cut_h - int(box_expand * height))

        far_left = int(np.argmin(landmarks[:, 0]))
        far_right = int(np.argmax(landmarks[:, 0]))

        left_up_point = np.int32([landmarks[far_left][0], cut_h])
        left_down_point = np.int32([landmarks[far_left][0], height])

        right_up_point = np.int32([landmarks[far_right][0], cut_h])
        right_down_point = np.int32([landmarks[far_right][0], height])

        cv2.drawContours(
            masks[i],
            np.int32(
                [
                    [
                        left_up_point,
                        left_down_point,
                        right_up_point,
                        right_down_point,
                        left_up_point,
                        right_up_point,
                        left_down_point,
                        right_down_point,
                    ]
                ]
            ),
            -1,
            color=(0),
            thickness=cv2.FILLED,
        )

        if show_nose:
            #  Calculate the bounding box coordinates for the nose
            nose_jaw_dist = (
                abs(landmarks[2][0] - landmarks[middle_nose_point][0]) * 0.10
            )  # 1, 15

            nose_min_x = int(landmarks[31][0] - nose_jaw_dist)
            nose_max_x = int(landmarks[35][0] + nose_jaw_dist)
            nose_min_y = cut_h
            nose_max_y = int(landmarks[nose, 1])

            # Clear the nose area from the mask using a rectangle
            mask_nose = np.ones((height, width), dtype=dtype)
            cv2.rectangle(
                mask_nose,
                (nose_min_x, nose_min_y),
                (nose_max_x, nose_max_y),
                color=(0),
                thickness=cv2.FILLED,
            )

            mask_nose = 1 - mask_nose
            masks[i] = masks[i] + mask_nose

    # If input was single frame, return single mask
    if landmark_list.shape[0] == 1:
        return masks[0]

    return 1 - torch.from_numpy(masks)
