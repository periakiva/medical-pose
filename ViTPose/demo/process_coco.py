import os
import json
import pandas as pd
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import os

import cv2
import numpy as np

def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))
def find_head(cate, location):
    flag = 0
    for i in range(7):
        v = location[i][2]
        if v == 2:
            flag = 1
        else:
            flag = 0
            break
    if flag == 1:
        x_min = min(location[0][0], location[1][0], location[2][0], location[3][0], location[4][0], location[5][0], location[6][0])
        x_max = max(location[0][0], location[1][0], location[2][0], location[3][0], location[4][0], location[5][0], location[6][0])

        y_min = min(location[0][1], location[1][1], location[2][1], location[3][1], location[4][1], location[5][1], location[6][1]) - 30
        y_max = max(location[0][1], location[1][1], location[2][1], location[3][1], location[4][1], location[5][1], location[6][1]) -15

        return True, [(x_min, y_min), (x_max, y_max)]
    else:
        return False, None

def find_torso(cate, location):
    flag = 0
    mark_keypoint = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    for i in mark_keypoint:
        v = location[i][2]
        if v == 2:
            flag = 1
        else:
            flag = 0
            break
    if flag == 1:
        x = [location[j][0] for j in mark_keypoint]
        x_min = min(x) -15
        x_max = max(x) + 15

        y = [location[j][1] for j in mark_keypoint]
        y_min = min(y) - 30
        y_max = max(y)

        return True, [(x_min, y_min), (x_max, y_max)]
    else:
        return False, None

dir = '/shared/group/coco/annotations/person_keypoints_val2017.json'


f = open(dir)
json_file = json.load(f)
# data = json.load(dir)

s = 1

imgs = json_file['images']
anns = json_file['annotations']

categories = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
num = 0
img_list = []
for ann in anns:
    # if num > 100:
    #     break
    img_id = int(ann['image_id'])
    img_id_ = "%012d" % img_id + '.jpg'

    keypoints = ann['keypoints']
    location = []
    for i, cate in enumerate(categories):
        x = keypoints[3 * i + 0]
        y = keypoints[3 * i + 1]
        v = keypoints[3 * i + 2]
        location.append([x, y, v])

    Head_exist, location = find_torso(categories, location)
    if Head_exist:
        if not img_id in img_list:
            img_list.append(img_id)
            prefix = '/shared/group/coco/val2017'
            _path = os.path.join(prefix, img_id_)


            I = cv2.imread(_path)



            I[location[0][1]:location[1][1],location[0][0]:location[1][0],:] = 0
            # I[], :, :] = 0
            I = rotate_bound(I, angle=110)





            target_image_dir = '/shared/niudt/pose_estimation/test_images/coco_legs_rotate110'
            if not os.path.exists(target_image_dir):
                os.makedirs(target_image_dir)
            save_name = os.path.join(target_image_dir, img_id_)
            cv2.imwrite(save_name, I)

            s = 1
            # num = num + 1






