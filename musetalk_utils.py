import os
import cv2
import torch
import numpy as np
from einops import rearrange
from PIL import Image, ImageDraw,ImageFilter
import scipy.ndimage

def pilimg_to_cv2img(pil_img):

    numpy_image = np.array(pil_img)
    
    # to 3 channels
    if numpy_image.ndim == 2:
        numpy_image = np.repeat(numpy_image[:, :, np.newaxis], 3, axis=2)
    
    # remove Alpha
    if numpy_image.shape[2] == 4:
        numpy_image = numpy_image[:, :, :3]
    
    # to BRG
    bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return bgr_image

def tensorimg_to_cv2img(tensor_img):
    numpy_image = tensor_img.numpy()
    numpy_image = numpy_image * 255.0
    numpy_image = numpy_image.astype('uint8')
    rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def cv2img_to_tensorimg(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    numpy_image = np.array(img_rgb)
    numpy_image = numpy_image / 255.0
    tensor_img = torch.from_numpy(numpy_image)
    return tensor_img

def pilimg_to_tensorimg(pil_img):
    numpy_image = np.array(pil_img)
    tensor_img = torch.tensor(numpy_image, dtype=torch.float32) / 255.0
    return tensor_img

def tensorimg_to_pilimg(tensor_img):
    numpy_image = (tensor_img * 255).byte().numpy()
    
    numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(numpy_image)
    
    return pil_img

def is_normalized(keypoints) -> bool:
    point_normalized = [
        0 <= np.abs(k[0]) <= 1 and 0 <= np.abs(k[1]) <= 1 
        for k in keypoints 
        if k is not None
    ]
    if not point_normalized:
        return False
    return np.all(point_normalized)


def get_half_face_mask(landmark, width, height):

    mask = Image.new("RGB", (width, height), (0,0,0))

    # https://www.researchgate.net/profile/Fabrizio-Falchi/publication/338048224/figure/fig1/AS:837860722741255@1576772971540/68-facial-landmarks.jpg
    points = landmark[0:17]
    # points.append(landmark[30])

    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=(255,255,255))

    return mask

def draw_landmarks(img, landmarks):
    
    img_copy = img.copy()
    for i, (x, y) in enumerate(landmarks):
        # # https://www.researchgate.net/profile/Fabrizio-Falchi/publication/338048224/figure/fig1/AS:837860722741255@1576772971540/68-facial-landmarks.jpg
        if i == 29 or i==48 or i == 54:
            # center nose , left mouth, right mouth
            cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)  # red
        else:
            cv2.circle(img_copy, (x, y), 2, (0, 255, 0), -1)  # green

    return img_copy


def get_landmards_by_posekey(pose_kps):
    # print("in get_landmards_by_posekey len(pose_kps)", len(pose_kps))
    land_marks = []
    for pose_frame in pose_kps:
        width, height = pose_frame["canvas_width"], pose_frame["canvas_height"]
        person_landmark = []
        for person in pose_frame["people"]:

            if "face_keypoints_2d" in person and person["face_keypoints_2d"] is not None:
            
                n = len(person["face_keypoints_2d"]) // 3

                facial_kps = rearrange(np.array(person["face_keypoints_2d"]), "(n c) -> n c", n=n, c=3)[:, :2]

                if is_normalized(facial_kps):
                    facial_kps *= (width, height)
                
                facial_kps = facial_kps.astype(np.int32)
                
                one_person_land_marks = [(x, y) for x, y in facial_kps]

                person_landmark.append(one_person_land_marks)
            else:
                print("not found face!!!")
        
        land_marks.append(person_landmark)
    
    return land_marks

def get_mouth_center_point_by_landmark(landmark):
    mouth_center_x = (landmark[51][0] + landmark[57][0]) // 2
    mouth_center_y = (landmark[51][1] + landmark[57][1]) // 2
    return (mouth_center_x, mouth_center_y)

def get_mouth_width_by_landmark(landmark):
    left_mouth_x = landmark[48][0]  # left mouth
    right_mouth_x = landmark[54][0]  # right mouth
    return right_mouth_x - left_mouth_x


def get_image_face_bbox(landmark):

    # face bbox
    left = min(landmark[i][0] for i in range(0, 17))
    right = max(landmark[i][0] for i in range(0, 17))
    bottom = max(landmark[i][1] for i in range(0, 27))

    # 51 top mouth
    # 57 bottom mouth
    # mouth_center_x = (landmark[51][0] + landmark[57][0]) // 2
    mouth_center_y = (landmark[51][1] + landmark[57][1]) // 2


    # left_mouth_x = landmark[48][0]  
    # right_mouth_x = landmark[54][0]

    # mouth_width = right_mouth_x - left_mouth_x

    # harf_x_left = mouth_center_x - left
    # harf_x_right = right - mouth_center_x
    # harf_x = max(harf_x_left, harf_x_right)

    # print("left:", bottom)
    # print("harf_x:", harf_x)

    # left = mouth_center_x - harf_x
    # right = mouth_center_x + harf_x

    one_fourth_y = bottom - mouth_center_y
    top = bottom - one_fourth_y*4

    # middle_y = bottom - landmark[29][1]
    # top = bottom - middle_y * 2


    # TODO，out-of-bounds process
    # top = top + top_reserve
    # bottom = bottom + bottom_reserve
    # left = left + left_reserve
    # right = right + right_reserve
    
    face_bbox = left, top, right, bottom

    return face_bbox

def get_face_center_point_and_rotate_angles(landmarks):
    
    landmarks = np.array(landmarks)

    # face center point
    center_point = np.mean(landmarks, axis=0)
    
    # left eye and right eye
    # left_point = landmarks[36]
    # right_point = landmarks[45]
    
    # left mouth and right mouth
    left_point = landmarks[48]
    right_point = landmarks[54]
    
    # cal angle
    angle = np.arctan2(right_point[1] - left_point[1], right_point[0] - left_point[0]) * 180 / np.pi
    
    return center_point, angle

def get_rotated_image(origin_image, face_center_point, rotate_angle):
        
    rotation_matrix = cv2.getRotationMatrix2D(tuple(face_center_point), rotate_angle, 1)
    rotated_image = cv2.warpAffine(origin_image, rotation_matrix, (origin_image.shape[1], origin_image.shape[0]), flags=cv2.INTER_NEAREST)

    return rotated_image


def get_rotatedimage_landmarks(landmark, face_center_point, rotate_angle):
    
    landmark = np.array(landmark)
    
    rotation_matrix = cv2.getRotationMatrix2D(tuple(face_center_point), rotate_angle, 1)

    adjusted_landmarks = landmark - face_center_point
    rotated_landmark = np.dot(rotation_matrix[:, :2], adjusted_landmarks.T).T + face_center_point

    converted_landmarks = [(int(point[0]), int(point[1])) for point in rotated_landmark]

    return converted_landmarks

def adjust_landmarks_to_crop(landmarks, bbox):

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    
    offset_x = left
    offset_y = top
    
    adjusted_landmarks = [(x - offset_x, y - offset_y) for x, y in landmarks]
    
    return adjusted_landmarks

def get_face_img_and_face_bbox(image, landmark, crop_type, top_reserve, bottom_reserve, left_reserve, right_reserve):

    # face bbox
    left = min(landmark[i][0] for i in range(0, 17))
    right = max(landmark[i][0] for i in range(0, 17))
    bottom = max(landmark[i][1] for i in range(0, 27))

    # modify top last
    bottom = bottom + bottom_reserve
    left = left - left_reserve
    right = right + right_reserve

    # mouth up center: 51
    # mouth down center: 57
    mouth_center_x = (landmark[51][0] + landmark[57][0]) // 2
    mouth_center_y = (landmark[51][1] + landmark[57][1]) // 2

    left_mouth_x = landmark[48][0]
    right_mouth_x = landmark[54][0]

    mouth_width = right_mouth_x - left_mouth_x

    harf_x_left = mouth_center_x - left
    harf_x_right = right - mouth_center_x

    if crop_type == "middle-min":
        harf_x = min(harf_x_left, harf_x_right)

        # print("left:", bottom)
        # print("harf_x:", harf_x)

        left = mouth_center_x - harf_x
        right = mouth_center_x + harf_x
    elif crop_type == "middle-max":
        harf_x = max(harf_x_left, harf_x_right)

        # print("left:", bottom)
        # print("harf_x:", harf_x)

        left = mouth_center_x - harf_x
        right = mouth_center_x + harf_x
    elif crop_type == "full":
        pass

    # left = int(mouth_center_x - mouth_width)
    # right = int(mouth_center_x + mouth_width)

    # one_fourth_height = bottom - mouth_center_y
    # half_height = one_fourth_height * 2

    # middle_y = bottom - half_height
    
    # if middle_y < landmark[28][1]:
    #     middle_y = landmark[28][1]
    # if middle_y > landmark[30][1]:
    #     middle_y = landmark[30][1]

    # half_height = bottom - middle_y
    # top = bottom - half_height * 2

    # landmark29 in v-center
    middle_y = bottom - landmark[29][1]
    top = bottom - middle_y * 2

    top = top - top_reserve
   
   # out of bounds
    left = max(0, left)
    top = max(0, top)
    right = min(image.shape[1], right)
    bottom = min(image.shape[0], bottom)

    # print(f"left: {left}, top: {top}, right: {right}, bottom: {bottom}")
    
    face_image = image[top:bottom, left:right]

    resized_face_image = cv2.resize(face_image,(256,256))
    
    face_bbox = left, top, right, bottom
    
    return face_image, resized_face_image, face_bbox


def create_uncrop_mask(width, height, center, v_axes, h_axes):


    mask = np.zeros((height, width), dtype=np.uint8)

    axes = (h_axes, v_axes)
    angle = 90
    color = 255

    cv2.ellipse(mask, center, axes, angle, 0, 360, color, thickness=-1)

    pil_image = Image.fromarray(mask)

    return pil_image
    


def uncrop_to_rotated_image(rotated_face, musetalk_face, rotated_bbox, rotated_image, uncrop_mask, extend, radius):

    mask = uncrop_mask.copy()

    # TODO，optimize
    mask = mask.convert('L')

    rotated_face_copy = rotated_face.copy()

    rotated_face_copy.paste(musetalk_face, (0, 0), mask)

    x_min, y_min, x_max, y_max = rotated_bbox

    origin_width, origin_height = rotated_image.size

    x_min = max(0, x_min)
    y_min = max(0, y_min)

    x_max = min(x_max, origin_width)
    y_max = min(y_max, origin_height)
    
    width = x_max - x_min
    height = y_max - y_min
    
    rotated_face_copy = rotated_face_copy.resize((width, height))

    # print("width:", width)
    # print("Height:", height)

    # musetalk_face = musetalk_face.resize((width, height))
    
    # mask = uncrop_mask

    # mask = mask.convert('L')

    # mask = mask.resize((width, height))

    if extend != 0:
        mask = expand_mask(mask, extend, True)

    if radius != 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=radius))

    # print(f"musetalk_face mode:{musetalk_face.mode} {musetalk_face.size}, rotated_image mode: {rotated_image.mode}, {rotated_image.size}, {mask.size}")

    rotated_image.paste(rotated_face_copy, (x_min, y_min))

    mask = mask.convert('RGB')

    return rotated_image, mask

def unrotated_image(musetalk_rotated_image, face_center_point, rotate_angle, width, height):
    rotation_matrix = cv2.getRotationMatrix2D(face_center_point, -rotate_angle, 1)
    musetalk_origin_image = cv2.warpAffine(musetalk_rotated_image, rotation_matrix, (width, height))
        
    return musetalk_origin_image

def expand_mask(mask, expand, tapered_corners):
    
    mask = np.array(mask)
    c = 0 if tapered_corners else 1

    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])

    iterations = abs(expand)

    operation = scipy.ndimage.morphology.binary_erosion if expand < 0 else scipy.ndimage.morphology.binary_dilation

    mask = operation(mask, structure=kernel, iterations=iterations)

    return Image.fromarray(mask.astype(np.uint8) * 255)


def blend_to_origin_image(origin_image, musetalk_origin_image, origin_face_mask, extend, radius):
    
    origin_face_mask = origin_face_mask.convert('L')

    origin_face_mask = expand_mask(origin_face_mask, extend, True)

    origin_face_mask = origin_face_mask.resize(musetalk_origin_image.size)

    origin_face_mask = origin_face_mask.filter(ImageFilter.BoxBlur(radius=radius))
    origin_face_mask = origin_face_mask.filter(ImageFilter.GaussianBlur(radius=radius))

    origin_image.paste(musetalk_origin_image, (0, 0), origin_face_mask)

    return origin_image, origin_face_mask