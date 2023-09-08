import os


kpts = ['nose',
        'mouth',
        'throat',
        'chest',
        'stomach',
        'left_upper_arm',
        'right_upper_arm',
        'left_lower_arm',
        'right_lower_arm',
        'left_wrist',
        'right_wrist',
        'left_hand',
        'right_hand',
        'left_upper_leg',
        'right_upper_leg',
        'left_knee',
        'right_knee',
        'left_lower_leg',
        'right_lower_leg',
        'left_foot',
        'right_foot']

skeleton = [[1, 2, (0 , 215, 255)], [2, 3, (0, 215, 255)], [3, 4, (0, 215, 255)], [4, 5, (0, 215, 255)], [3, 6, (255, 0, 0)], [3, 7, (0, 255, 0)], [6,8, (255, 0, 0)], [7,9, (0, 255, 0)], [8, 10, (255, 0, 0)], [9, 11, (0, 255, 0)], [10, 12, (255, 0, 0)], [11, 13, (0, 255, 0)], [5, 14, (255, 0, 0)], [5, 15, (0, 255, 0)], [14, 16, (255, 0, 0)], [15, 17, (0, 255, 0)], [16, 18, (255, 0, 0)], [17, 19,(0, 255, 0)], [18, 20, (255, 0, 0)], [19, 21, (0, 255, 0)]]

skeleton_info = {}
for i, skeleton_ in enumerate(skeleton):
    a_index = skeleton_[0] - 1
    b_index = skeleton_[1] - 1
    color = skeleton_[2]
    a = kpts[a_index]
    b = kpts[b_index]
    temp = dict(link= (a, b), id = i, color = [color[0], color[1], color[2]])
    # temp['link'] = (a, b)
    # temp['id'] = i
    # temp['color'] = [color[0], color[1], color[2]]
    skeleton_info[i] = temp

print(skeleton_info)

print('num=', len([
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ]))
import numpy as np
print(list(np.ones(21)))

sigma = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]

aver_sigma = sum(sigma)/17
print('number of sigma = ', len(sigma))



print(list(np.ones(21) * 0.067))