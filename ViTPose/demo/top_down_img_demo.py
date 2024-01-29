# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
from glob import glob
from xtcocotools.coco import COCO
import tqdm
import json
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
import kwcoco
# from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
#                          vis_pose_result)
import numpy as np
from mmpose.datasets import DatasetInfo

def dictionary_contents(path: str, types: list, recursive: bool = False) -> list:
    """
    Extract files of specified types  from directories, optionally recursively.

    Parameters:
        path (str): Root directory path.
        types (list): List of file types (extensions) to be extracted.
        recursive (bool, optional): Search for files in subsequent directories if True. Default is False.

    Returns:
        list: List of file paths with full paths.
    """
    files = []
    if recursive:
        path = path + "/**/*"
    for type in types:
        if recursive:
            for x in glob(path + type, recursive=True):
                files.append(os.path.join(path, x))
        else:
            for x in glob(path + type):
                files.append(os.path.join(path, x))
    return files

#/shared/niudt/pose_estimation/vitpose/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_medic_256x192.py
# /shared/niudt/pose_estimation/vitpose/ViTPose/tools/medic_train/epoch_50.pth
def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('--pose_config', default='/shared/niudt/pose_estimation/vitpose/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/medic/vitbase_patient.py', help='Config file for detection')
    parser.add_argument('--pose_checkpoint', default='/home/local/KHQ/peri.akiva/projects/medical-pose/ViTPose/weights/pose_model.pth', help='Checkpoint file')
    parser.add_argument('--img-root', type=str,
                        default='/shared/niudt/pose_estimation/vitpose/ViTPose/demo_videos/input/M2-6/images/',
                        help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='/home/local/KHQ/peri.akiva/projects/medical-pose/bbox_detection_results/bbox_detections.json',
        help='Json file containing image info.')
    
    parser.add_argument(
        '--out-json-file',
        type=str,
        default='/home/local/KHQ/peri.akiva/projects/medical-pose/bbox_detection_results/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_with_patient_dets_with_pose.mscoco.json',
        help='Json file containing image info.')
    
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='./test_results_m2-6/',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.8, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=10,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')

    args = parser.parse_args()
    save_intermediate = True
    assert args.show or (args.out_img_root != '')

    # coco = COCO(args.json_file)
    coco = kwcoco.CocoDataset(args.json_file)

    keypoints_cats = [
                        "nose", "mouth", "throat","chest","stomach","left_upper_arm",
                        "right_upper_arm","left_lower_arm","right_lower_arm","left_wrist",
                        "right_wrist","left_hand","right_hand","left_upper_leg",
                        "right_upper_leg","left_knee","right_knee","left_lower_leg", 
                        "right_lower_leg", "left_foot", "right_foot", "back"
                    ]
    
    keypoints_cats_dset = [{'name': value, 'id': index} for index, value in enumerate(keypoints_cats)]
    coco.dataset['keypoint_categories'] = keypoints_cats_dset
    
    # print(coco.dataset)
    
    print(f"coco annotations: {len(coco.anns)}")
    # build the pose model from a config file and a checkpoint file
    # print(f"pose config: {args.pose_config}")
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # paths = dictionary_contents(args.img_root, types=['*.jpg'], recursive=True)
    # imageid_to_path = {}
    # for path in tqdm.tqdm(paths, total=len(paths)):
    #     # print(f"path {path}")
    #     image_id = "_".join(path.split('/')[-2:])
    #     # print(f"image_id {image_id}")
    #     if image_id in imageid_to_path.keys():
    #         print(f"image_id {image_id} already in dictionary")
    #     imageid_to_path[image_id] = path
    
    img_keys = list(coco.imgs.keys())
    # print(f"img_keys: {img_keys[:10]}")
    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # process each image
    pbar = tqdm.tqdm(enumerate(coco.imgs.items()), total=len(list(coco.imgs.keys())))
    original_num_annots = len(coco.anns)
    for index, (image_id, img_dict) in pbar:
        
        # if i > 40:
        #     continue
        
        # get bounding box annotations
        path = img_dict['file_name']
        # image_id = img_keys[i]
        image = coco.load_image(image_id)#[0]
        h, w, c = image.shape
        # print(f"image shape: {image.shape}")
        # image_name = os.path.join(args.img_root, image['file_name'])
        # image_name = imageid_to_path[image['file_name']]
        image_name = path
        # print(image_name)
        # print(f"filename: {image['file_name']}")
        # ann_ids = coco.getAnnIds(image_id)
        ann_ids = coco.index.gid_to_aids[image_id]

        # print(f"image id: {image_id}, image_name: {image_name}")
        # make person bounding boxes
        person_results = []
        # print(f"annotations for image: {len(ann_ids)}")
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            if 'label' in ann.keys():
                if ann['label'] != 'patient':
                    continue
            else:
                continue
            
            # print(f"ann: {ann}")
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            cat_id = ann['category_id']
            # bbox_score = ann['bbox_score']
            bbox_label = ann['label']
            person['label'] = bbox_label
            person_results.append(person)

        
        # print(f"person results length: {len(person_results)}")
        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if len(person_results)>0:
            
            # print(pose_results)
            
            pose_keypoints = pose_results[0]['keypoints'].tolist()
            bbox = pose_results[0]['bbox'].tolist()
            pose_keypoints_list = []
            
            for index, keypoint in enumerate(pose_keypoints):
                kp_dict = {'xy': [keypoint[0], keypoint[1]], 
                           'keypoint_category_id': index, 
                           'keypoint_category': keypoints_cats[index]}
                pose_keypoints_list.append(kp_dict)
            
            keypoints_ann = {}
            keypoints_ann['image_id'] = image_id
            # keypoints_ann['left_hand_offset'] = 
            # keypoints_ann['right_hand_offset'] = 
            # keypoints_ann['object_offset'] = 
            keypoints_ann['category_id'] = cat_id
            keypoints_ann['label'] = 'patient'
            keypoints_ann['bbox'] = bbox
            keypoints_ann['keypoints'] = pose_keypoints_list
            
            
            
            # print(keypoints_ann)
            
            ann_ids = coco.index.gid_to_aids[image_id]
            for ann_id in ann_ids:
                ann = coco.index.anns[ann_id]
                # print(ann)
                image_center = w//2
                if ann['category_id'] == 0:
                    bx, by, bw, bh = ann['bbox']
                    hcx, hcy = bx+bw//2, by+by//w
                    hand_point = np.array((hcx, hcy))
                    offset_vector = []
                    for joint in pose_keypoints:
                        jx, jy, jc = joint
                        joint_point = np.array((jx, jy))
                        dist = np.linalg.norm(joint_point - hand_point)
                        offset_vector.append(dist)
                        
                    # print(f"offset vector: {offset_vector}")
                    if hcx <= image_center:
                    # #     # hcx, hcy = bx+bw//2, by+by//w
                        keypoints_ann['left_hand_offset'] = offset_vector
                    elif hcx > image_center:
                        keypoints_ann['right_hand_offset'] = offset_vector
                elif ann['category_id'] in [1,2,3,4,5,6,7,8,9,10,11]:
                    bx, by, bw, bh = ann['bbox']
                    ocx, ocy = bx+bw//2, by+by//w
                    object_point = np.array((ocx, ocy))
                    offset_vector = []
                    for joint in pose_keypoints:
                        jx, jy, jc = joint
                        joint_point = np.array((jx, jy))
                        dist = np.linalg.norm(joint_point - object_point)
                        offset_vector.append(dist)
                    keypoints_ann['object_offset'] = offset_vector
                    keypoints_ann['object_offset_wrt'] = ann_id


            coco.add_annotation(**keypoints_ann)
            # print(ann_ids)
            # for ann_id in ann_ids:
            # ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            ### add hands to joint offset vectors
            ### add object to joint offst vectors
            # print(f"ann: {ann}")
            # print(f"pose results: {pose_results}")
            
            # coco.anns[ann_id]['keypoints'] = pose_results[0]['keypoints'].tolist()
            
            
            
                
                # person_results.append(person)
        # person['keypoints'] = 
        pbar.set_description(f"Anns dataset: {len(coco.dataset['annotations'])}, orginal coco.ann: {original_num_annots}")
        # print(f"pose results: {pose_results[0]['keypoints'].tolist()}")
        # print(f"coco ann: {coco.anns[ann_id]}")
        # print(f"coco type: {type(coco)}")
        # print(f"coco dataset: {coco.dataset.keys()}")
        
        # print(f"pose length: {pose_results[0]['keypoints'].shape}")
        # exit()
    
        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            current_name = image_name.split('/')[-1]
            # current_name = image['file_name']
            # out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')
            out_file = os.path.join(args.out_img_root, current_name)

        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)
        
        if save_intermediate:
            if index % 5000 == 1:
                coco.dump(args.out_json_file, newlines=True)
                print(f"saved intermediate at index {index}")

        
        
    print(f"coco.dataset type: {type(coco.dataset)}")
    # with open(args.out_json_file, 'w') as json_file: 
    #     json.dump(coco.dataset, json_file) 
    coco.dump(args.out_json_file, newlines=True)


if __name__ == '__main__':
    
    """
    python ./ViTPose/demo/top_down_img_demo.py --json-file /home/local/KHQ/peri.akiva/projects/medical-pose/bbox_detection_results/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_with_patient_dets.mscoco.json --pose_config ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_medic_casualty_256x192.py --out-img-root /data/datasets/ptg/m2_tourniquet/output_imgs/pose --out-json-file /home/local/KHQ/peri.akiva/projects/medical-pose/ViTPose/results/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_with_patient_dets_with_pose.mscoco.json


    """
    
    main()
