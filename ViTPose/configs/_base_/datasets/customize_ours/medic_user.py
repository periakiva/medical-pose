dataset_info = dict(
    dataset_name='medic_user',
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
    keypoint_info={ #'left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm', 'left_hand', 'right_hand
        0:
        dict(name='left_upper_arm',
             id=0,
             color=[0, 0, 139],
             type='upper',
             swap=''),
        1:
        dict(
            name='right_upper_arm',
            id=1,
            color=[255, 112, 132],
            type='upper',
            swap=''),
        2:
        dict(
            name='left_lower_arm',
            id=2,
            color=[0, 0, 139],
            type='upper',
            swap=''),
        3:
        dict(
            name='right_lower_arm',
            id=3,
            color=[255, 112, 132],
            type='upper',
            swap=''),
        4:
        dict(
            name='left_hand',
            id=4,
            color=[0, 0, 139],
            type='upper',
            swap=''),
        5:
        dict(
            name='right_hand',
            id=5,
            color=[255, 112, 132],
            type='upper',
            swap='')
    },
    skeleton_info={
        0:
            {'link': ('right_hand', 'right_lower_arm'), 'id': 0, 'color':[255, 112, 132]},
        1:
            {'link': ('right_upper_arm', 'right_lower_arm'), 'id': 1, 'color': [255, 112, 132]},
        2:
            {'link': ('left_upper_arm', 'left_lower_arm'), 'id': 2, 'color': [0, 0, 139]},
        3:
            {'link': ('left_lower_arm', 'left_hand'), 'id': 3, 'color': [0, 0, 139]}
    },

    joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    sigmas=[0.67, 0.67, 0.67, 0.67, 0.67, 0.67]
)