import cv2
import os
import numpy as np


def re_order(input_list):
    new_list = []
    num_list = []
    for img in input_list:
        num = int(img.split('_')[-1].split('.')[0])
        num_list.append(num)

    arr = np.array(num_list)  

    index = np.argsort(arr) 

    for i in range(len(input_list)):
        new_list.append(input_list[index[i]])

    return new_list

if __name__ == '__main__':
    img_path = '/shared/niudt/DATASET/Medical/Maydemo/2023-4-25/selected_videos/new/vitpose/M1-57'
    img_list = os.listdir(img_path)
    img_list = re_order(input_list=img_list)


  
    video = cv2.VideoWriter('/shared/niudt/DATASET/Medical/Maydemo/2023-4-25/selected_videos/new/vitpose/M1-57.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
                            (1280, 720))

    for i in range(len(img_list)):
        img = cv2.imread(os.path.join(img_path, img_list[i]))  
        # print(img.shape)
        # img = cv2.resize(img,(1981,991))
        video.write(img)  

    video.release()

    #ffmpeg -r 10 -i %5.jpg /shared/niudt/gbo/detectron2/demo/tracking_dataset_video_version/output.mp4
