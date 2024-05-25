
import cv2
from einops import rearrange
import torch

from . import musetalk_utils
from . import musetalk_global_data

import comfy

class MuseTalkPreprocess:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "origin_images": ("IMAGE",),
                "pose_kps": ("POSE_KEYPOINT",),
                "crop_type": (["full", "middle-min", "middle-max"],),
                "top_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "bottom_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "left_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "right_reserve": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", 
                    # "FACE_BBOX", "IMAGE", "FACE_CENTER_POINT", "ROTATE_ANGLE", "FACE_BBOX", "IMAGE", "LANDMARK", 
                    "IMAGE", )
    RETURN_NAMES = (
        "rotated_faces",
        # "rotated_bboxs",
        # "rotated_images",
        # "face_center_points",
        # "rotated_angles",
        # "origin_face_bboxs",
        # "origin_face_masks",
        # "landmarks",
        "rotated_faces_with_landmarks"
    )

    FUNCTION = "preprocess"
    CATEGORY = "MuseTalkUtils"

    def preprocess(self, origin_images, pose_kps, crop_type, top_reserve, bottom_reserve, left_reserve, right_reserve):

        print(f"MuseTalkPreprocess preprocess, len(origin_images): {len(origin_images)}")
        
        global rotated_faces
        global rotated_faces_with_landmarks

        global rotated_bboxs
        global rotated_images
        global face_center_points
        global rotated_angles
        global origin_face_bboxs
        global origin_face_masks
        global origin_face_landmarks


        musetalk_global_data.rotated_faces = []
        musetalk_global_data.rotated_faces_with_landmarks = []

        musetalk_global_data.rotated_bboxs = []
        musetalk_global_data.rotated_images = []

        musetalk_global_data.face_center_points = []
        musetalk_global_data.rotated_angles = []
        musetalk_global_data.origin_face_bboxs = []
        musetalk_global_data.origin_face_masks = []
        musetalk_global_data.rotated_resized_half_face_masks = []
        

        if len(origin_images) != len(pose_kps):
            print("origin_images is not same with pose_kps by len")
            return None

        musetalk_global_data.origin_face_landmarks = musetalk_utils.get_landmards_by_posekey(pose_kps)

        idx = -1

        pbar = comfy.utils.ProgressBar(len(origin_images))

        for image, landmark in zip(origin_images, musetalk_global_data.origin_face_landmarks):

            idx = idx + 1

            # print("landmark len: ", len(landmark))
            # print("landmark: ", landmark)

            if len(landmark) == 0:
                if len(musetalk_global_data.rotated_faces) > 0:
                    cur_index = len(musetalk_global_data.rotated_faces)-1
                    musetalk_global_data.rotated_faces.append(musetalk_global_data.rotated_faces[cur_index])
                    musetalk_global_data.rotated_faces_with_landmarks.append(musetalk_global_data.rotated_faces_with_landmarks[cur_index])

                    musetalk_global_data.rotated_bboxs.append(musetalk_global_data.rotated_bboxs[cur_index])
                    musetalk_global_data.rotated_images.append(musetalk_global_data.rotated_images[cur_index])
                    musetalk_global_data.face_center_points.append(musetalk_global_data.face_center_points[cur_index])
                    musetalk_global_data.rotated_angles.append(musetalk_global_data.rotated_angles[cur_index])
                    musetalk_global_data.origin_face_bboxs.append(musetalk_global_data.origin_face_bboxs[cur_index])
                    musetalk_global_data.origin_face_masks.append(musetalk_global_data.origin_face_masks[cur_index])
                    musetalk_global_data.origin_face_landmarks.append(musetalk_global_data.origin_face_landmarks[cur_index])
                    print(f"not found face, image index: {idx}")
                    continue
                else:
                    # TODO: process no face in first frame
                    continue

            landmark = landmark[0]
            
            origin_image = musetalk_utils.tensorimg_to_cv2img(image)
            origin_height, origin_width = image.shape[:2]
            # print("origin_image shape: ", image.shape)

            origin_face_bbox = musetalk_utils.get_image_face_bbox(landmark)
            musetalk_global_data.origin_face_bboxs.append(origin_face_bbox)
            # print("origin_face_bbox: ", origin_face_bbox)

            origin_face_mask = musetalk_utils.get_half_face_mask(landmark, origin_width, origin_height)
            musetalk_global_data.origin_face_masks.append(musetalk_utils.pilimg_to_tensorimg(origin_face_mask))
            # print("origin_face_mask: ", origin_face_mask.size)

            face_center_point, rotate_angle = musetalk_utils.get_face_center_point_and_rotate_angles(landmark)
            musetalk_global_data.face_center_points.append(face_center_point)
            musetalk_global_data.rotated_angles.append(rotate_angle)

            # print("face_center_point: ", face_center_point)
            # print("rotate_angle: ", rotate_angle)

            rotated_image = musetalk_utils.get_rotated_image(origin_image, face_center_point, rotate_angle)
            musetalk_global_data.rotated_images.append(musetalk_utils.cv2img_to_tensorimg(rotated_image))
            # print("rotated_image: ", rotated_image)

            rotated_landmark = musetalk_utils.get_rotatedimage_landmarks(landmark, face_center_point, rotate_angle)

            # print("rotated_landmark:",rotated_landmark)

            rotated_face, resized_rotated_face, rotated_face_bbox = musetalk_utils.get_face_img_and_face_bbox(rotated_image, rotated_landmark, crop_type, top_reserve, bottom_reserve, left_reserve, right_reserve)

            rotated_face_landmark = musetalk_utils.adjust_landmarks_to_crop(rotated_landmark, rotated_face_bbox)

            left, top, right, bottom = rotated_face_bbox

            # print(rotated_face_bbox, right - left, top - bottom)

            rotated_resized_half_face_mask = musetalk_utils.get_half_face_mask(rotated_face_landmark, right - left, bottom - top)

            rotated_resized_half_face_mask = rotated_resized_half_face_mask.resize((256, 256))

            musetalk_global_data.rotated_resized_half_face_masks.append(rotated_resized_half_face_mask)

            rotated_face_with_landmark = musetalk_utils.draw_landmarks(rotated_face, rotated_face_landmark)

            rotated_face_with_landmark = cv2.resize(rotated_face_with_landmark, (256, 256))

            width = rotated_face_with_landmark.shape[1]
            height = rotated_face_with_landmark.shape[0]
            
            cv2.line(rotated_face_with_landmark, (width//2, 0), (width//2, height-1), (255, 0, 0), 1)  # v-center-line, blue
            cv2.line(rotated_face_with_landmark, (0, height//2), (width-1, height//2), (255, 0, 0), 1)  # h-center-line, blue

            musetalk_global_data.rotated_faces_with_landmarks.append(musetalk_utils.cv2img_to_tensorimg(rotated_face_with_landmark))

            musetalk_global_data.rotated_bboxs.append(rotated_face_bbox)
            musetalk_global_data.rotated_faces.append(musetalk_utils.cv2img_to_tensorimg(resized_rotated_face))

            pbar.update(1)

        return (
                torch.stack(musetalk_global_data.rotated_faces, dim=0), 
                # rotated_bboxs, 
                # torch.stack(rotated_images, dim=0), 
                # face_center_points, 
                # rotated_angles, 
                # origin_face_bboxs, 
                # torch.stack(origin_face_masks, dim=0), 
                # origin_face_landmarks,
                torch.stack(musetalk_global_data.rotated_faces_with_landmarks, dim=0), 
                )     



if __name__ == "__main__":
    musetalk_utils.get_landmards_by_posekey(None)
    print("hello")
