
import cv2
import os


def videoToPNG(videoPath, pngPath):
    '''
    视频提取图片
    '''
    if not os.path.exists(videoPath):
        print("文件不存在:", videoPath)
        return
    file_name = os.path.basename(videoPath)
    folder_name = os.path.join(pngPath, file_name.split('.')[0])
    # 创建输出文件夹
    try:
        os.makedirs(folder_name, exist_ok=True)
    except Exception as e:
        print("文件夹创建失败：", folder_name, e)
    vc = cv2.VideoCapture(videoPath)
    count = 0
    rval = vc.isOpened()
    while rval:
        rval, frame = vc.read()
        if not rval:
            break
        pic_path = folder_name + "/" + str(count) + ".jpg"
        # 这里修改了导出的大小
        # img = cv2.resize(frame, (640, 480))
        # 这里将所有的帧都导出了，如果文件比较大的情况下会比较多，这里可以根据自己的需求做一些限制。
        cv2.imwrite(pic_path, frame)
        count += 1
    vc.release()
    print(videoPath, "读取完成")


# 视频路径，这里使用的是相对路径
filePath = "/shared/niudt/DATASET/Medical/Maydemo/2023-4-25/selected_videos/new/bbox_detections/bmw"
# 导出图片路径
pngFolder = "/shared/niudt/DATASET/Medical/Maydemo/2023-4-25/selected_videos/new/bbox_detections/bmw/results"
fileList = os.listdir(filePath)
for fileName in fileList:
    videoPath = os.path.join(filePath, fileName)
    videoToPNG(videoPath, pngFolder)

# video_path = '/shared/niudt/DATASET/Medical/Maydemo/2023-4-25/selected_videos/new/bbox_detections/bmw/bmw100.mp4'