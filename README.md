# Medical-Partial-Body-Pose-Estimation

Our model includes two stages: patient detector and pose estimator. To use the model, please follow the below instructions.
# First Stage: Patient Detector
Because the pose prediction of the patient needs the proposal (bounding box) as input, so we need to run run our trained patient detector to get these bounding boxes first. Following the following steps to run the detector.
## Installation

### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional but needed by demo and visualization


### Build detectron2 
Step 1: Install Pytroch: following the instruction in https://pytorch.org/ to install the latest version of pytorch.

Step 2: Following the corresponding structure, clone the code, and run:
```
cd ./Medical-Partial-Body-Pose-Estimation/

python -m pip install -e detectron2
```

### Install other dependencies
In order to make the model compatible to your syste, you may need adjust the version of some pachages:

```
pip install pillow==9.5
pip install opencv-python
```

### Download the pre-trained model

Download our [model](https://drive.google.com/file/d/1OHAr31n41keDTJygDmFfOgsXwpriuFT9/view?usp=sharing) and save it to ```./detectron2/weights``` folder

## Run the inference of your images
The model takes input as input, if you have video, you should first split the video as images and save it to some place.

Feel free to use our prepared data for test. You can download them at [test_data](https://drive.google.com/file/d/1mOwxB5doD-zhMsQkKte2Gt8V40oxR7PN/view?usp=sharing)

Then get the detection results by running:

```
python ./detectron2/demo/bbox_detection_medic.py --config-file configs/medic_pose/medic_pose.yaml --input you_path/*.jpg
```


## The results you will get by running the model

You will get a dict including the frame level preditions, with the structure of


```
├── demo
│   ├── bbox_detection_results
│   │   ├── vis
│   │     └── frame1_result.jpg
|   |     └── frame2_result.jpg
|   |     └── ......
│   ├── bbox_detections.json

```

You will also get frame-level prediction is ``vis`` folder (see the expample of the following Figure) and a json file named ``bbox_detections.json`` for the sebsequent pose estimation.

<img src="description/bbox_pred_examples.jpg" width="300" >

