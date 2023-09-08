checkpoint_config = dict(interval=10)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

log_level = 'INFO'
load_from = '/shared/niudt/pose_estimation/vitpose/ViTPose/pretrained_model/vitpose-b.pth'
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
