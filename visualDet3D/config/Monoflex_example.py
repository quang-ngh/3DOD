from easydict import EasyDict as edict
import os 
import numpy as np

cfg = edict()
cfg.obj_types = ['Car', 'Pedestrian', 'Cyclist']
cfg.anchor_prior = False
## trainer
trainer = edict(
    gpu = 0,
    max_epochs = 200,
    disp_iter = 50,
    save_iter = 40,
    test_iter = 40,
    training_func = "train_rtm3d",
    test_func = "test_mono_detection",
    evaluate_func = "evaluate_kitti_obj",
)

cfg.trainer = trainer

## path
path = edict()
path.data_path = "data/svm_kitti/training"
path.test_path = "data/svm_kitti/testing"
path.visualDet3D_path = "visualDet3D/visualDet3D"
path.project_path = "visualDet3D/results"

if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)
path.project_path = os.path.join(path.project_path, 'MonoFlex_SVM')
if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)

path.log_path = os.path.join(path.project_path, "log")
if not os.path.isdir(path.log_path):
    os.mkdir(path.log_path)

path.checkpoint_path = os.path.join(path.project_path, "checkpoint")
if not os.path.isdir(path.checkpoint_path):
    os.mkdir(path.checkpoint_path)

path.preprocessed_path = os.path.join(path.project_path, "output")
if not os.path.isdir(path.preprocessed_path):
    os.mkdir(path.preprocessed_path)

path.train_imdb_path = os.path.join(path.preprocessed_path, "training")
if not os.path.isdir(path.train_imdb_path):
    os.mkdir(path.train_imdb_path)

path.val_imdb_path = os.path.join(path.preprocessed_path, "validation")
if not os.path.isdir(path.val_imdb_path):
    os.mkdir(path.val_imdb_path)

cfg.path = path

## optimizer
optimizer = edict(
    type_name = 'adamw',
    keywords = edict(
        lr        = 1e-4,
        weight_decay = 0,
    ),
    clipped_gradient_norm = 35.0
)
cfg.optimizer = optimizer
## scheduler
scheduler = edict(
    type_name = 'MultiStepLR',
    keywords = edict(
        milestones = [90, 120]
    )
)
cfg.scheduler = scheduler

## data
data = edict(
    batch_size = 8,
    num_workers = 4,
    rgb_shape = (384, 1280, 3),
    train_dataset = "KittiMonoFlexDataset",
    val_dataset   = "KittiMonoDataset",
    test_dataset  = "KittiMonoTestDataset",
    train_split_file = os.path.join(cfg.path.visualDet3D_path, 'data', 'kitti', 'chen_split', 'train.txt'),
    val_split_file   = os.path.join(cfg.path.visualDet3D_path, 'data', 'kitti', 'chen_split', 'val.txt'),
    max_occlusion = 999,
    min_z = -999,
)

data.augmentation = edict(
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
    #crop_top = 100,
)
data.train_augmentation = [
    edict(type_name='RandomWarpAffine', keywords=edict(output_w=data.augmentation.cropSize[1], output_h=data.augmentation.cropSize[0])),
    edict(type_name='ConvertToFloat'),
    edict(type_name="Shuffle", keywords=edict(
            aug_list=[
                edict(type_name="RandomBrightness", keywords=edict(distort_prob=1.0)),
                edict(type_name="RandomContrast", keywords=edict(distort_prob=1.0, lower=0.6, upper=1.4)),
                edict(type_name="Compose", keywords=edict(
                   aug_list=[
                       edict(type_name="ConvertColor", keywords=edict(transform='HSV')),
                       edict(type_name="RandomSaturation", keywords=edict(distort_prob=1.0, lower=0.6, upper=1.4)),
                       edict(type_name="ConvertColor", keywords=edict(current='HSV', transform='RGB')),
                   ] 
                ))
            ]
        )
    ),
    edict(type_name='RandomMirror', keywords=edict(mirror_prob=0.5)),
    edict(type_name="FilterObject"),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
data.test_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
cfg.data = data

## networks
detector = edict()
detector.obj_types = cfg.obj_types
detector.name = 'MonoFlex'
detector.backbone = edict(
    name='dlanet',
    depth=34,
    out_indices=(0, 1, 2, 3, 4, 5),
    pretrained=None,
    #name='resnet',
    #depth=18,
    #out_indices=(3, ),
    #pretrained=True,
    #norm_eval=False,
)
head_loss = edict(
    gamma=2.0,
    output_w = data.rgb_shape[1] / 4.0
)
head_test = edict(
    score_thr=0.1,
)

head_layer = edict(
    #input_features=256,
    #head_features=64,
    input_features=64,
    head_features=256,
    head_dict={'hm': len(cfg.obj_types), 'bbox2d': 4, 'hps': 20,
               'rot': 8, 'dim': 3, 'reg': 2, 'depth': 1,
               "depth_uncertainty": 1, "corner_uncertainty": 3}
)
detector.head = edict(
    num_classes     = len(cfg.obj_types),
    num_joints      = 9,
    max_objects     = 32,
    layer_cfg       = head_layer,
    loss_cfg        = head_loss,
    test_cfg        = head_test
)
detector.loss = head_loss
cfg.detector = detector
