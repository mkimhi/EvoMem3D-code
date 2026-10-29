point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bbox_depth=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.625),
            final_dim=(320, 800),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='PETRFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'prev_exists'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
            'centers2d', 'depths', 'prev_exists', 'lidar2img', 'intrinsics',
            'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose',
            'ego_pose_inv'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'scene_token', 'gt_bboxes_3d',
                   'gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.625),
            final_dim=(320, 800),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=False),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv'
                ],
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'lidar2img', 'intrinsics', 'extrinsics',
                    'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CustomNuScenesDataset',
        data_root='./data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes2d_temporal_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=True,
                with_label=True,
                with_bbox_depth=True),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=True),
            dict(
                type='GlobalRotScaleTransImage',
                rot_range=[-0.3925, 0.3925],
                translation_std=[0, 0, 0],
                scale_ratio_range=[0.95, 1.05],
                reverse_angle=True,
                training=True),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='PETRFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv', 'prev_exists'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes',
                    'gt_labels', 'centers2d', 'depths', 'prev_exists',
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'scene_token',
                           'gt_bboxes_3d', 'gt_labels_3d'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        num_frame_losses=1,
        seq_split_num=2,
        seq_mode=True,
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'img', 'prev_exists',
            'img_metas'
        ],
        queue_length=5,
        use_valid_flag=True,
        filter_empty_gt=False),
    val=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes2d_temporal_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv'
                        ],
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'img', 'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv'
                        ],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'scene_token'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'img', 'img_metas'
        ],
        queue_length=1),
    test=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes2d_temporal_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv'
                        ],
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'img', 'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv'
                        ],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'scene_token'))
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'img', 'img_metas'
        ],
        queue_length=1),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=42192,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='ResizeCropFlipRotImage',
            data_aug_conf=dict(
                resize_lim=(0.47, 0.625),
                final_dim=(320, 800),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                H=900,
                W=1600,
                rand_flip=True),
            training=False),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='PETRFormatBundle3D',
                    collect_keys=[
                        'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                        'img_timestamp', 'ego_pose', 'ego_pose_inv'
                    ],
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'img', 'lidar2img', 'intrinsics', 'extrinsics',
                        'timestamp', 'img_timestamp', 'ego_pose',
                        'ego_pose_inv'
                    ],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'pad_shape', 'scale_factor', 'flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'scene_token'))
            ])
    ])
checkpoint_config = dict(interval=1758, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/tm_evomem3d'
load_from = 'ckpts/repdetr3d_eva02_baseline.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_gpus = 8
batch_size = 2
num_iters_per_epoch = 1758
num_epochs = 24
queue_length = 1
num_frame_losses = 1
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
    'ego_pose', 'ego_pose_inv'
]
sim_fpn = dict(
    scale_factors=[4, 2, 1, 0.5],
    in_channels=1024,
    out_channels=256,
    out_indices=[2, 3, 4, 5])
model = dict(
    type='RepDetr3D',
    num_frame_head_grads=1,
    num_frame_backbone_grads=1,
    num_frame_losses=1,
    use_grid_mask=True,
    stride=[4, 8, 16, 32],
    position_level=[0, 1, 2, 3],
    img_backbone=dict(
        type='EVAViT',
        img_size=320,
        patch_size=16,
        window_size=16,
        global_window_size=20,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=2.6666666666666665,
        window_block_indexes=[
            0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22
        ],
        sim_fpn=dict(
            scale_factors=[4, 2, 1, 0.5],
            in_channels=1024,
            out_channels=256,
            out_indices=[2, 3, 4, 5]),
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cp=True,
        flash_attn=True),
    img_roi_head=dict(
        type='YOLOXHeadCustom',
        num_classes=10,
        in_channels=256,
        strides=[4, 8, 16, 32],
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))),
    pts_bbox_head=dict(
        type='SparseHead',
        num_classes=10,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        with_dn=True,
        with_ego_pos=True,
        match_with_velo=False,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                embed_dims=256,
                num_layers=6,
                transformerlayers=dict(
                    type='Detr3DTemporalDecoderLayer',
                    batch_first=True,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='DeformableFeatureAggregationCuda',
                            embed_dims=256,
                            num_groups=8,
                            num_levels=4,
                            num_cams=6,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))),
    temporal_memory=dict(
        enabled=True,
        detector_adapter='repdetr3d',
        keyframe_interval=5,
        max_frames_between_full=4,
        policy=dict(
            type='mlp',
            checkpoint=None,
            load_weights=False,
            feature_names=[
                'frame_number', 'info_frame_idx', 'is_scene_start', 'dt_s',
                'ego_x_m', 'ego_y_m', 'ego_z_m', 'ego_yaw_deg',
                'delta_ego_translation_m', 'ego_speed_mps', 'ego_accel_mps2',
                'yaw_rate_deg_s', 'abs_yaw_rate_deg_s', 'yaw_accel_deg_s2',
                'abs_yaw_accel_deg_s2', 'num_annotations',
                'avg_object_distance_m', 'median_object_distance_m',
                'min_object_distance_m', 'p10_object_distance_m',
                'p90_object_distance_m', 'near_object_count_10m',
                'near_object_count_20m', 'near_object_count_30m',
                'near_object_count_50m', 'avg_box_length_m', 'avg_box_width_m',
                'avg_box_height_m', 'avg_box_volume_m3', 'avg_box_area_m2',
                'avg_object_speed_mps', 'max_object_speed_mps',
                'camera_annotation_count', 'max_camera_annotation_count',
                'min_camera_annotation_count', 'avg_camera_annotation_count',
                'avg_2d_box_area_px', 'max_2d_box_area_px',
                'new_object_proxy_count', 'matched_object_proxy_count',
                'disappeared_object_proxy_count', 'new_object_proxy_fraction'
            ],
            mean=[
                19.567702, 19.567702, 0.024921083, 0.48656002, 1182.1858,
                1385.3081, 0, 31.97956, 2.5787889, 5.1699474, 0.27622647,
                -0.098026916, 2.1476363, -0.004204151, 1.6220828, 26.958465,
                32.142464, 29.066243, 8.9008939, 13.750016, 54.315544,
                2.7509553, 9.0727696, 14.954644, 22.62934, 3.6646713, 1.686735,
                1.80034, 19.724399, 7.8472459, 1.5351453, 8.1423581, 37.788503,
                13.818242, 1.352218, 6.2980838, 38678.796, 365563.36,
                6.2972255, 20.661239, 5.7041037, 0.2693361
            ],
            scale=[
                11.587536, 11.587536, 0.15588465, 0.082544361, 562.12445,
                585.10477, 1, 104.69116, 1.79063, 3.5718243, 2.0322777,
                5.0668897, 4.5902765, 3.6931669, 3.3178829, 17.85341,
                8.5337187, 9.6403387, 6.7434273, 7.2861665, 15.116219, 3.56017,
                8.7497763, 12.383638, 16.34568, 1.6049316, 0.43034936,
                0.36658377, 16.581536, 4.4716831, 1.5116474, 4.127657,
                25.722243, 8.6701681, 2.153647, 4.2870405, 39275.125, 380484.6,
                5.9729865, 15.675711, 4.6237755, 0.21515531
            ],
            hidden_dim=32,
            labels=('easy', 'medium', 'hard', 'very_hard'),
            actions=dict(easy='L8', medium='L12', hard='L16', very_hard='L24'),
            fallback_action='L24'),
        policy_supervision=dict(
            enabled=True,
            target='similarity',
            warmup_ratio=0.1,
            max_train_steps=42192,
            thresholds=(0.9, 0.76, 0.675),
            class_weights=(1.0, 1.0, 1.0, 3.0),
            weight=1.0,
            feature_noise_std=0.1),
        updater='gated_residual',
        shared_memory_gate=True,
        partial_layers=(8, 12, 16),
        updater_cfg=dict(
            channels=256, feature_format='spatial', residual_scale_init=0.0),
        partial_layer='block8',
        projector_cfg=dict(
            in_channels=1024,
            out_channels=256,
            scale_factors=(4, 2, 1, 0.5),
            out_indices=(2, 3, 4, 5)),
        memory_is_pre_pe=True,
        pe_mode='separate_recompute',
        freeze_pretrained_detector=True,
        use_ego_conditioning=False,
        use_distillation=False,
        distill_weight=0.0,
        train_with_teacher=False,
        inference_teacher_forbidden=True,
        policy_offline_features='teacher_forcing',
        policy_offline_csv=None,
        adaptive_reset=dict(
            enabled=True,
            max_frames_between_full=2,
            ego_speed_mps=2.0,
            max_object_speed_mps=5.0)),
    tm_debug_shapes=False,
    tm_debug_latency=False,
    tm_debug_compare_full_path=False,
    tm_assert_no_teacher_in_inference=True,
    pretrained=None)
ida_aug_conf = dict(
    resize_lim=(0.47, 0.625),
    final_dim=(320, 800),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
find_unused_parameters = False
runner = dict(type='IterBasedRunner', max_iters=42192)
_online_policy_features = [
    'frame_number', 'info_frame_idx', 'is_scene_start', 'dt_s', 'ego_x_m',
    'ego_y_m', 'ego_z_m', 'ego_yaw_deg', 'delta_ego_translation_m',
    'ego_speed_mps', 'ego_accel_mps2', 'yaw_rate_deg_s', 'abs_yaw_rate_deg_s',
    'yaw_accel_deg_s2', 'abs_yaw_accel_deg_s2'
]
_online_policy_mean = [
    19.567702, 19.567702, 0.024921083, 0.48656002, 1182.1858, 1385.3081, 0.0,
    31.97956, 2.5787889, 5.1699474, 0.27622647, -0.098026916, 2.1476363,
    -0.004204151, 1.6220828
]
_online_policy_scale = [
    11.587536, 11.587536, 0.15588465, 0.082544361, 562.12445, 585.10477, 1.0,
    104.69116, 1.79063, 3.5718243, 2.0322777, 5.0668897, 4.5902765, 3.6931669,
    3.3178829
]
_online_offline_policy_features = [
    'frame_number', 'info_frame_idx', 'is_scene_start', 'dt_s', 'ego_x_m',
    'ego_y_m', 'ego_z_m', 'ego_yaw_deg', 'delta_ego_translation_m',
    'ego_speed_mps', 'ego_accel_mps2', 'yaw_rate_deg_s', 'abs_yaw_rate_deg_s',
    'yaw_accel_deg_s2', 'abs_yaw_accel_deg_s2', 'num_annotations',
    'avg_object_distance_m', 'median_object_distance_m',
    'min_object_distance_m', 'p10_object_distance_m', 'p90_object_distance_m',
    'near_object_count_10m', 'near_object_count_20m', 'near_object_count_30m',
    'near_object_count_50m', 'avg_box_length_m', 'avg_box_width_m',
    'avg_box_height_m', 'avg_box_volume_m3', 'avg_box_area_m2',
    'avg_object_speed_mps', 'max_object_speed_mps', 'camera_annotation_count',
    'max_camera_annotation_count', 'min_camera_annotation_count',
    'avg_camera_annotation_count', 'avg_2d_box_area_px', 'max_2d_box_area_px',
    'new_object_proxy_count', 'matched_object_proxy_count',
    'disappeared_object_proxy_count', 'new_object_proxy_fraction'
]
_online_offline_policy_mean = [
    19.567702, 19.567702, 0.024921083, 0.48656002, 1182.1858, 1385.3081, 0,
    31.97956, 2.5787889, 5.1699474, 0.27622647, -0.098026916, 2.1476363,
    -0.004204151, 1.6220828, 26.958465, 32.142464, 29.066243, 8.9008939,
    13.750016, 54.315544, 2.7509553, 9.0727696, 14.954644, 22.62934, 3.6646713,
    1.686735, 1.80034, 19.724399, 7.8472459, 1.5351453, 8.1423581, 37.788503,
    13.818242, 1.352218, 6.2980838, 38678.796, 365563.36, 6.2972255, 20.661239,
    5.7041037, 0.2693361
]
_online_offline_policy_scale = [
    11.587536, 11.587536, 0.15588465, 0.082544361, 562.12445, 585.10477, 1,
    104.69116, 1.79063, 3.5718243, 2.0322777, 5.0668897, 4.5902765, 3.6931669,
    3.3178829, 17.85341, 8.5337187, 9.6403387, 6.7434273, 7.2861665, 15.116219,
    3.56017, 8.7497763, 12.383638, 16.34568, 1.6049316, 0.43034936, 0.36658377,
    16.581536, 4.4716831, 1.5116474, 4.127657, 25.722243, 8.6701681, 2.153647,
    4.2870405, 39275.125, 380484.6, 5.9729865, 15.675711, 4.6237755, 0.21515531
]
gpu_ids = range(0, 2)
