# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
from glob import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import json
from xtcocotools.coco import COCO

from predictor import VisualizationDemo

import warnings
warnings.filterwarnings("ignore")

# constants
WINDOW_NAME = "COCO detections"

def dictionary_contents(path: str, types: list, recursive: bool = False) -> list:
    """
    Extract files of specified types from directories, optionally recursively.

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

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


"""
python detectron2/demo/bbox_detections_medic.py --config detectron2/configs/medic_pose/medic_pose.yaml --input /data/datasets/ptg/m2_tourniquet/imgs/M2-65/*.jpg
"""

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/shared/niudt/Kitware/Medical-Partial-Body-Pose-Estimation/detectron2/configs/medic_pose/medic_pose.yaml",
        metavar="FILE",
        help="path to config file",
    )
    
    parser.add_argument(
        "--coco-file",
        default="/data/PTG/medical/object_anns/old_bbn/M2_Tourniquet/results/m2_with_lab_cleaned_fixed_data_with_steps_RESULTS/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_copy.mscoco.json",
        metavar="FILE",
        help="path to coco file",
    )
    
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file."
                        # , default='/shared/niudt/detectron2/images/Videos/k2/4.MP4'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'"
        , default= ['/shared/niudt/DATASET/Medical/Maydemo/2023-4-25/selected_videos/new/M2-16/*.jpg'] # please change here to the path where you put the images
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window."
        , default='./bbox_detection_results'
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,

        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)


    demo = VisualizationDemo(cfg)

    # json_file = {}
    # json_file['images'] = []
    # json_file['annotations'] = []
    # json_file['categories']  = []

    # # get new categorie
    # temp_bbox = {}
    # temp_bbox['id'] = 1
    # temp_bbox['name'] = 'patient'
    # temp_bbox['instances_count'] = 0
    # temp_bbox['def'] = ''
    # temp_bbox['synonyms'] = ['patient']
    # temp_bbox['image_count'] = 0
    # temp_bbox['frequency'] = ''
    # temp_bbox['synset'] = ''

    # json_file['categories'].append(temp_bbox)

    # temp_bbox = {}
    # temp_bbox['id'] = 2
    # temp_bbox['name'] = 'user'
    # temp_bbox['instances_count'] = 0
    # temp_bbox['def'] = ''
    # temp_bbox['synonyms'] = ['user']
    # temp_bbox['image_count'] = 0
    # temp_bbox['frequency'] = ''
    # temp_bbox['synset'] = ''

    # json_file['categories'].append(temp_bbox)

    coco = COCO(args.coco_file)
    patient_cat = {'id':41, 'name': 'patient'}
    user_cat = {'id':42, 'name': 'user'}
    coco.cats[patient_cat['id']] = patient_cat
    coco.cats[user_cat['id']] = user_cat
    
    anns_ids = list(coco.anns.keys())
    # print(anns_ids)
    
    # exit()
    

    
    # print(f"coco: {coco.imgs.keys()}")
    # print(f"coco: {coco.cats}")
    # print(f"coco: {coco.imgs}")
    # exit()
    # if len(args.input) == 1:
    #     args.input = glob.glob(os.path.expanduser(args.input[0]))
    #     assert args.input, "The input path(s) was not found"
    # paths = dictionary_contents(args.input[0], types=['*.JPG', '*.jpg', '*.JPEG', '*.jpeg'], recursive=True)
    
    ann_id = anns_ids[-1]+1
    num_img = 0
    pbar = tqdm.tqdm(coco.imgs.items())
    for img_id, img_dict in pbar:
        # use PIL, to be consistent with evaluation
        # print(img_id)
        # print(value)
        # exit()
        
        path = img_dict['file_name']
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        instances = predictions["instances"].to('cpu')
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

        boxes = boxes.tensor.detach().numpy()
        scores = scores.numpy()
        
        file_name = "_".join(path.split('/')[-2:])
        # print(f"file name: {file_name}")
        
        if boxes is not None:
            # num_img = num_img + 1


            # add images info
            # current_img = {}
            # # current_img['file_name'] = path.split('/')[-1]
            # current_img['file_name'] = file_name
            # current_img['id'] = num_img
            # current_img['height'] = 720
            # current_img['width'] = 1280
            # json_file['images'].append(current_img)

            for box_id, _bbox in enumerate(boxes):


                # add annotations
                current_ann = {}
                current_ann['id'] = ann_id
                current_ann['image_id'] = img_id
                current_ann['bbox'] = np.asarray(_bbox).tolist()#_bbox
                current_ann['category_id'] = 41
                current_ann['label'] = 'patient'
                current_ann['bbox_score'] = str(round(scores[box_id] * 100,2)) + '%'

                # if current_ann['category_id'] == 2:
                #     continue
                coco.anns[ann_id] = current_ann
                ann_id += 1



        vis_save_path = os.path.join(args.output, 'vis_results')
        os.makedirs(vis_save_path, exist_ok=True)
        # out_filename = os.path.join(vis_save_path, os.path.basename(path))
        out_filename = f"{vis_save_path}/{file_name}"
        # print(f'vis_save_path: {vis_save_path}')
        # print(f'out_filename: {out_filename}')
        if num_img % 30 == 1:
            visualized_output.save(out_filename)

    det_save_root = f"{args.output}/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_with_patient_dets.mscoco.json"
    
    # det_save_root = os.path.join(args.output, 'bbox_detections.json')
    with open(det_save_root, 'w') as fp:
        json.dump(coco, fp)

"""
python detectron2/demo/bbox_detections_medic.py --config-file detectron2/configs/medic_pose/medic_pose.yaml --input /data/datasets/ptg/m2_tourniquet/imgs
"""