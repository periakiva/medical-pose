import json


def main():
    pose_json = "/home/local/KHQ/peri.akiva/projects/Medical-Partial-Body-Pose-Estimation/ViTPose/results/pose_keypoints.json"
    with open(pose_json, 'r') as openfile:
        json_object = json.load(openfile)
    
    ann_sample = json_object['annotations'][0]
    
    print(ann_sample)


if __name__ == '__main__':
    main()