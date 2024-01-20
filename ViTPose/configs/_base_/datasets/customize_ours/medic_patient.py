dataset_info = dict(
    dataset_name='coco_medic',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='nose',
             id=0,
             color=[0, 215, 255],
             type='upper',
             swap=''),
        1:
        dict(
            name='mouth',
            id=1,
            color=[0, 215, 255],
            type='upper',
            swap=''),
        2:
        dict(
            name='throat',
            id=2,
            color=[0, 215, 255],
            type='upper',
            swap=''),
        3:
        dict(
            name='chest',
            id=3,
            color=[0, 215, 255],
            type='upper',
            swap=''),
        4:
        dict(
            name='stomach',
            id=4,
            color=[0, 215, 255],
            type='upper',
            swap=''),
        5:
        dict(
            name='left_upper_arm',
            id=5,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        6:
        dict(
            name='right_upper_arm',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        7:
        dict(
            name='left_lower_arm',
            id=7,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        8:
        dict(
            name='right_lower_arm',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        11:
        dict(
            name='left_hand',
            id=11,
            color=[255, 0, 0],
            type='upper',
            swap=''),
        12:
        dict(
            name='right_hand',
            id=12,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        13:
        dict(
            name='left_upper_leg',
            id=13,
            color=[255, 0, 0],
            type='lower',
            swap=''),

        14:
        dict(
            name='right_upper_leg',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        15:
        dict(
            name='left_knee',
            id=15,
            color=[255, 0, 0],
            type='lower',
            swap=''),
        16:
        dict(
            name='right_knee',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        17:
        dict(
            name='left_lower_leg',
            id=17,
            color=[255, 0, 0],
            type='lower',
            swap=''),
        18:
        dict(
            name='right_lower_leg',
            id=18,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        19:
        dict(
            name= 'left_foot',
            id=19,
            color=[255, 0, 0],
            type='lower',
            swap=''),
        20:
        dict(
            name='right_foot',
            id=20,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        21:
        dict(
            name='back',
            id=21,
            color=[147, 20, 255],
            type='lower',
            swap=''),
    },
    skeleton_info={
        0:
            {'link': ('nose', 'mouth'), 'id': 0, 'color': [0, 215, 255]},
        1:
            {'link': ('mouth', 'throat'), 'id': 1, 'color': [0, 215, 255]},
        2:
            {'link': ('throat', 'chest'), 'id': 2, 'color': [0, 215, 255]},
        3:
            {'link': ('chest', 'stomach'), 'id': 3, 'color': [0, 215, 255]},
        4:
            {'link': ('throat', 'left_upper_arm'), 'id': 4, 'color': [255, 0, 0]},
        5:
            {'link': ('throat', 'right_upper_arm'), 'id': 5, 'color': [0, 255, 0]},
        6:
            {'link': ('left_upper_arm', 'left_lower_arm'), 'id': 6, 'color': [255, 0, 0]},
        7:
            {'link': ('right_upper_arm', 'right_lower_arm'), 'id': 7, 'color': [0, 255, 0]},
        8:
            {'link': ('left_lower_arm', 'left_wrist'), 'id': 8, 'color': [255, 0, 0]},
        9:
            {'link': ('right_lower_arm', 'right_wrist'), 'id': 9, 'color': [0, 255, 0]},
        10:
            {'link': ('left_wrist', 'left_hand'), 'id': 10, 'color': [255, 0, 0]},
        11:
            {'link': ('right_wrist', 'right_hand'), 'id': 11, 'color': [0, 255, 0]},
        12:
            {'link': ('stomach', 'left_upper_leg'), 'id': 12, 'color': [255, 0, 0]},
        13:
            {'link': ('stomach', 'right_upper_leg'), 'id': 13, 'color': [0, 255, 0]},
        14:
            {'link': ('left_upper_leg', 'left_knee'), 'id': 14, 'color': [255, 0, 0]},
        15:
            {'link': ('right_upper_leg', 'right_knee'), 'id': 15, 'color': [0, 255, 0]},
        16:
            {'link': ('left_knee', 'left_lower_leg'), 'id': 16, 'color': [255, 0, 0]},
        17:
            {'link': ('right_knee', 'right_lower_leg'), 'id': 17, 'color': [0, 255, 0]},
        18:
            {'link': ('left_lower_leg', 'left_foot'), 'id': 18, 'color': [255, 0, 0]},
        19:
            {'link': ('right_lower_leg', 'right_foot'), 'id': 19, 'color': [0, 255, 0]},
        20:
            {'link': ('right_upper_leg', 'back'), 'id': 20, 'color': [147, 20, 255]},
        21:
            {'link': ('left_upper_leg', 'back'), 'id': 21, 'color': [147, 20, 255]},
    },

    joint_weights=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    sigmas=[
        0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067]
)