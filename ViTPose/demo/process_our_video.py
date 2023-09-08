# coding=utf-8

import os
import cv2

# the dir including videos you want to process
videos_src_path = '/shared/niudt/pose_estimation/Fabricate_dataset/our_video'
# the save path
videos_save_path = '/shared/niudt/pose_estimation/Fabricate_dataset/our_video_img'

videos = os.listdir(videos_src_path)
# videos = filter(lambda x: x.endswith('MP4'), videos)

for each_video in videos:
    print('Video Name :', each_video)
    # get the name of each video, and make the directory to save frames
    each_video_name, _ = each_video.split('.')
    # os.mkdir(videos_save_path + '/' + each_video_name)

    each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

    # get the full path of each video, which will open the video tp extract frames
    each_video_full_path = os.path.join(videos_src_path, each_video)

    cap = cv2.VideoCapture(each_video_full_path)

    frame_count = 1

    frame_rate = 1
    success = True
    # 计数
    num = 0
    while (success):
        success, frame = cap.read()
        if success == True:
            if not os.path.exists(each_video_save_full_path + each_video_name):
                os.makedirs(each_video_save_full_path + each_video_name)

            if frame_count % frame_rate == 0:
                cv2.imwrite(each_video_save_full_path + each_video_name + '/'+ "%06d.jpg" % num, frame)
                num += 1

        frame_count = frame_count + 1
    print('Final frame:', num)

