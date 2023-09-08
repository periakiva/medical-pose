import os


kpts = ['left_upper_arm',
        'right_upper_arm',
        'left_lower_arm',
        'right_lower_arm',
        'left_hand',
        'right_hand']

skeleton = [[0, 2, (255, 0, 0)], [2, 4,(255, 0, 0)], [1, 3, (0, 255, 0)], [3, 5, (0, 255, 0)]]

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
print(list(np.ones(6)))

sigma = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]

aver_sigma = sum(sigma)/17
print('number of sigma = ', len(sigma))



print(list(np.ones(6) * 0.67))