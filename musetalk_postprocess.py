
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import comfy
import time

from . import musetalk_utils
from . import musetalk_global_data


# def create_uncrop_mask(width, height, center, v_axes, h_axes):
class MuseTalkUncropMask:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 256, "min": -9999, "max": 9999, "step": 1}),
                "height": ("INT", {"default": 256, "min": -9999, "max": 9999, "step": 1}),
                "ellipse_center_x": ("INT", {"default": 128, "min": -9999, "max": 9999, "step": 1}),
                "ellipse_center_y": ("INT", {"default": 192, "min": -9999, "max": 9999, "step": 1}),
                "ellipse_center_v_axes": ("INT", {"default": 128, "min": -9999, "max": 9999, "step": 1}),
                "ellipse_center_h_axes": ("INT", {"default": 64, "min": -9999, "max": 9999, "step": 1}),
            },
        }
# top_reserve, bottom_reserve, left_reserve, right_reserve
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = (
        "images",
    )

    FUNCTION = "run"
    CATEGORY = "MuseTalkUtils"

    def run(self, width, height, ellipse_center_x, ellipse_center_y, ellipse_center_v_axes, ellipse_center_h_axes):
        pil_image_mask = musetalk_utils.create_uncrop_mask(width, height, (ellipse_center_x, ellipse_center_y), ellipse_center_v_axes, ellipse_center_h_axes)
        image = pil_image_mask.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image, )


class MuseTalkPostprocess:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "origin_images": ("IMAGE",),
                "musetalk_faces": ("IMAGE",),
                # "rotated_bboxs": ("FACE_BBOX",),
                # "rotated_images": ("IMAGE",),
                # "face_center_points": ("FACE_CENTER_POINT",),
                # "rotated_angles": ("ROTATE_ANGLE",),
                # "origin_face_bboxs": ("FACE_BBOX",),
                # "origin_face_masks": ("IMAGE",),
                # "landmarks":("LANDMARK",),
                # "uncrop_mask":("IMAGE",),
                "extend": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "blur_radius": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "extend1": ("INT", {"default": -5, "min": -9999, "max": 9999, "step": 1}),
                "blur_radius1": ("INT", {"default": 5, "min": -9999, "max": 9999, "step": 1}),
            },
            "optional": {
                "uncrop_mask":("IMAGE",),
            }
        }
# top_reserve, bottom_reserve, left_reserve, right_reserve
    RETURN_TYPES = ("IMAGE",
                    # "IMAGE", "IMAGE","IMAGE",
                    )
    RETURN_NAMES = (
        "images",
        # "uncrop_masks",
        # "uncroped_images",
        # "face_masks",
        )

    FUNCTION = "postprocess"
    CATEGORY = "MuseTalkUtils"

    def getRealIndex(self, index, origin_img_len):
        if index >= origin_img_len: 
            return (origin_img_len * 2 - index - 1)
        else:
            return index

    def postprocess(self, origin_images, musetalk_faces, 
                    # rotated_bboxs, rotated_images, 
                    # face_center_points, rotated_angles, origin_face_bboxs, origin_face_masks, landmarks,
                    # uncrop_mask,
                    extend, blur_radius,extend1, blur_radius1,
                    uncrop_mask = None):
        
        global rotated_faces
        global rotated_faces_with_landmarks

        global rotated_bboxs
        global rotated_images
        global face_center_points
        global rotated_angles
        global origin_face_bboxs
        global origin_face_masks
        global origin_face_landmarks
        global rotated_resized_half_face_masks

        if uncrop_mask is not None:
            uncrop_mask = uncrop_mask[0]

        print(f"MuseTalkPreprocess postprocess, len(origin_images): {len(origin_images)}, len(musetalk_faces): {len(musetalk_faces)}")

        musetalk_face_image_count = len(musetalk_faces)

        
        # TODO, process default value

        origin_img_len = len(origin_images)
        rotated_bboxs_len = len(musetalk_global_data.rotated_bboxs)
        rotated_images_len = len(musetalk_global_data.rotated_images)
        face_center_points_len = len(musetalk_global_data.face_center_points)
        rotated_angles_len = len(musetalk_global_data.rotated_angles)
        origin_face_bboxs_len = len(musetalk_global_data.origin_face_bboxs)
        origin_face_masks_len = len(musetalk_global_data.origin_face_masks)

        MAX_LEN = origin_img_len * 2

        print(f"origin_img_len: {origin_img_len}, rotated_bboxs_len: {rotated_bboxs_len}, rotated_images_len: {rotated_images_len}, face_center_points_len: {face_center_points_len}, rotated_angles_len: {rotated_angles_len},origin_face_bboxs_len: {origin_face_bboxs_len}, origin_face_masks_len: {origin_face_masks_len}")
        if origin_img_len == rotated_bboxs_len == rotated_images_len == face_center_points_len == rotated_angles_len == origin_face_bboxs_len ==origin_face_masks_len:
            if origin_img_len < musetalk_face_image_count:
                pass
        else:
            print("the len is not same")
            return (None)

        result_images = []

        # face_masks = []
        # uncrop_masks = []
        # uncroped_images = []
        idx = 0

        pbar = comfy.utils.ProgressBar(len(musetalk_faces))

        for musetalk_face in musetalk_faces:

            start_time0 = time.time()

            real_index = self.getRealIndex(idx, origin_img_len)
            # TODO: valid real_index
            
            origin_image = origin_images[real_index]

            rotated_bbox = musetalk_global_data.rotated_bboxs[real_index]
            rotated_image = musetalk_global_data.rotated_images[real_index]
            face_center_point = musetalk_global_data.face_center_points[real_index]
            rotate_angle = musetalk_global_data.rotated_angles[real_index]
            # origin_face_bbox = musetalk_global_data.origin_face_bboxs[real_index]
            origin_face_mask = musetalk_global_data.origin_face_masks[real_index]
            # landmark = musetalk_global_data.origin_face_landmarks[real_index]
            rotated_face = musetalk_global_data.rotated_faces[real_index]

            
            # musetalk_face = musetalk_utils.tensorimg_to_cv2img(musetalk_face)
            # rotated_image = musetalk_utils.tensorimg_to_cv2img(rotated_image)
            
            origin_image_height, origin_image_width = musetalk_utils.tensorimg_to_cv2img(origin_image).shape[:2]

            # print("origin_image shape: ", origin_image.shape[:2])
            # print("musetalk_face shape: ", musetalk_face.shape[:2])
            # print("rotated_image shape: ", rotated_image.shape[:2])


            if uncrop_mask == None:

                pil_uncrop_mask_image = musetalk_global_data.rotated_resized_half_face_masks[real_index]

                start_time1 = time.time()
                musetalk_rotated_image, pil_uncrop_mask_image = musetalk_utils.uncrop_to_rotated_image(musetalk_utils.tensorimg_to_pilimg(rotated_face), 
                                                                                musetalk_utils.tensorimg_to_pilimg(musetalk_face), 
                                                                                    rotated_bbox, 
                                                                                    musetalk_utils.tensorimg_to_pilimg(rotated_image), 
                                                                                    pil_uncrop_mask_image, extend, blur_radius)
                # uncrop_masks.append(musetalk_utils.pilimg_to_tensorimg(pil_uncrop_mask_image))

                print(f"frame index: {idx}, real_index: {real_index}, uncrop one frame, use: {((time.time() - start_time1)*1000):.2f} ms")
            else:
                musetalk_rotated_image, _ = musetalk_utils.uncrop_to_rotated_image(musetalk_utils.tensorimg_to_pilimg(rotated_face), 
                                                                                musetalk_utils.tensorimg_to_pilimg(musetalk_face), 
                                                                                    rotated_bbox, 
                                                                                    musetalk_utils.tensorimg_to_pilimg(rotated_image), 
                                                                                    musetalk_utils.tensorimg_to_pilimg(uncrop_mask), 0, 0)
                # uncrop_masks.append(uncrop_mask)
            
            musetalk_rotated_image_tensor = musetalk_utils.pilimg_to_tensorimg(musetalk_rotated_image)
            # uncroped_images.append(musetalk_rotated_image_tensor)
            
            # uncrop_masks.append(musetalk_utils.pilimg_to_tensorimg(uncrop_mask))

            # TODO: optimize
            musetalk_origin_image = musetalk_utils.unrotated_image(musetalk_utils.tensorimg_to_cv2img(musetalk_rotated_image_tensor), face_center_point, rotate_angle, origin_image_width, origin_image_height)

            musetalk_origin_image_tensor = musetalk_utils.cv2img_to_tensorimg(musetalk_origin_image)

            # landmark = landmark[0]

            # (origin_image, musetalk_origin_image, origin_face_bbox, mouth_center_point, mouth_width, origin_face_mask, extend, radius):
            
            start_time1 = time.time()
            result_image, face_mask = musetalk_utils.blend_to_origin_image(musetalk_utils.tensorimg_to_pilimg(origin_image), 
                                                                musetalk_utils.tensorimg_to_pilimg(musetalk_origin_image_tensor), 
                                                                musetalk_utils.tensorimg_to_pilimg(origin_face_mask),
                                                                extend1, blur_radius1)
            
            print(f"frame index: {idx}, real_index: {real_index}, blend one frame, use: {((time.time() - start_time1)*1000):.2f} ms")

            result_images.append(musetalk_utils.pilimg_to_tensorimg(result_image))
            # face_masks.append(musetalk_utils.pilimg_to_tensorimg(face_mask))

            pbar.update(1)

            print(f"frame index: {idx}, real_index: {real_index}, processed one frame, total use: {((time.time() - start_time0)*1000):.2f} ms")

            idx = (idx + 1)%MAX_LEN

        return (
                torch.stack(result_images, dim=0), 
                # torch.stack(uncrop_masks, dim=0), 
                # torch.stack(uncroped_images, dim=0), 
                # torch.stack(face_masks, dim=0), 
                )



if __name__ == "__main__":

    # def postprocess(self, origin_images, 
    # musetalk_faces, 
    # rotated_bboxs, 
    # rotated_images, 
    # face_center_points, 
    # rotated_angles, 
    # origin_face_bboxs, 
    # origin_face_masks, 
    # landmarks):

    ori_img = Image.open("./ori.png")
    ori_img_cv2 = cv2.cvtColor(np.array(ori_img), cv2.COLOR_RGB2BGR)
    ori_img_tensor = musetalk_utils.cv2img_to_tensorimg(ori_img_cv2)    

    musetalk_face = Image.open("./failed_image_musetalk_face36.jpg")
    musetalk_face_cv2 = cv2.cvtColor(np.array(musetalk_face), cv2.COLOR_RGB2BGR)
    musetalk_face_tensor = musetalk_utils.cv2img_to_tensorimg(musetalk_face_cv2)

    rotated_bbox = (150, 201, 721, 729)

    rotated_image = Image.open("./failed_image_rotated_image36.jpg")
    rotated_image_cv2 = cv2.cvtColor(np.array(rotated_image), cv2.COLOR_RGB2BGR)
    rotated_image_tensor = musetalk_utils.cv2img_to_tensorimg(rotated_image_cv2)

    face_center_point = [(50,50)]

    rotated_angle = [20]

    origin_face_bbox = [(0,0,500,500)]

    origin_face_mask = [None]

    isok, musetalk_rotated_image = musetalk_utils.uncrop_to_rotated_image(musetalk_utils.tensorimg_to_cv2img(musetalk_face_tensor), 
                                                                          rotated_bbox, 
                                                                          musetalk_utils.tensorimg_to_cv2img(rotated_image_tensor))
    print(isok)



    # test = MuseTalkPostprocess()
    # test.postprocess()

