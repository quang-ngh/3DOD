import os
from tqdm import tqdm
from easydict import EasyDict
from typing import Sized, Sequence
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from visualDet3D.networks.utils.registry import PIPELINE_DICT
from visualDet3D.evaluator.kitti.evaluate import evaluate
from visualDet3D.evaluator.kitti_depth_prediction.evaluate_depth import evaluate_depth
from visualDet3D.networks.utils.utils import BBox3dProjector, BackProjection
from visualDet3D.data.kitti.utils import write_result_to_file
from visualDet3D.networks.lib.fast_utils.hill_climbing import post_opt
from visualDet3D.utils.utils import draw_3D_box
from visualDet3D.utils.bbox3d import Box3D
import json

def read_extrinsic(): 
        return (
            np.array([
                [-0.015964651480317116,          0.99987155199050903,            0.0014207595959305763],
                [0.58267718553543091,          0.010458142496645451,            -0.81263637542724609],
                [-0.81254684925079346,            -0.012145611457526684,            -0.58276933431625366]
                ]),
            # np.array([ -0.037088312208652496, 1.0999122858047485, -0.9999545629739761353  ])
            np.array([ -0.037088312208652496, 1.5999122858047485, -1.2545629739761353])

        )
def read_intrinsic(): 
        return ( 
            np.array([[304.007121, 0.0, 638.469054], [0.0, 304.078429, 399.956311], [0.0, 0.0, 1.0]]),
            np.array([0.138281, 0.025172, -0.030963, 0.005019]) 
        )

(matK, matD) = read_intrinsic()
(matR, vecT) = read_extrinsic()
# vecT_svm = np.array([-1.5999122858047485, -0.037088312208652496, -1.2545629739761353])        #   old vecT
vecT_svm = np.array([-1.96434001e+00,  1.64003220e-03,  -5.78054998e-01])     #   new vecT: camera is on the right of origin
# vecT = np.array([0.0392197, 1.59137,  -0.574618])
vec_T_reshape = vecT.reshape((3,1))
matR_inv = np.linalg.inv(matR)
extrinsic_mat = np.hstack([matR, vec_T_reshape])

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def read_json(fl: str):
    obj = None
    with open(fl, mode= 'r') as reader:
        obj = json.load(reader)
    reader.close()
    assert obj != None
    return obj

def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def get_ground_truth(json_file, annotations):
    """
    Return the image and Box3D prediction from annotation
    args:
        ann: json file
    """

    basename, ext = json_file.split('.')
    
    rect_corners = []
    obj = None

    obj = read_json(f'{annotations}/{basename}.json')

    for item in obj['labels']:  
        depth = item['box3d']['location']['x']

        if abs(depth) < 5 or abs(depth) > 20:
            continue
        box3d = Box3D(item, true_z = False)
        coor = box3d.in_camera_coordinate(axis = 'rotz') 
        
        rect_corners.append(coor)
        # print(box3d)
    
    # breakpoint()
    if len(rect_corners) > 0:
        rect_corners = np.stack(rect_corners, axis = 0)

    return rect_corners

def draw_gt_bev(img, corners_3d, bev_size, scale_factor):
    
    H, W, C = img.shape
    thickness = 3
    color = [0,0,0]
    size = (384, 384)
    
    if len(corners_3d) == 0:
        return

    for idx, corner in enumerate(corners_3d): 
        # bev_box[:, 0] = bev_box[:, 0]*(-1)
        """
        bev_box: [depth, width] --> for visualization it need to be 
        transform to [width, depth]
        """ 
        bev_box = corner[:2, :].T
        bev_box[:, 0] = size[1]/scale_factor[0] + bev_box[:, 0] * size[1]/bev_size[0]   #   Width
        bev_box[:, 1] = size[0]/scale_factor[1] + bev_box[:, 1] * size[0]/bev_size[1]   #   Depth 
        bev_box = bev_box[:, ::-1] 
        # print(bev_box)
        # breakpoint() 
        bev_box = [tuple(x) for x in bev_box.astype('int32').tolist()]
        
        # color = np.random.randint(0, 235, [1, 3]).tolist()[0]
        # color = colors[idx]
        color = [0,0,0]
        cv2.line(img, bev_box[0], bev_box[1], color, thickness)
        cv2.line(img, bev_box[1], bev_box[2], color, thickness)
        # breakpoint()
        cv2.line(img, bev_box[2], bev_box[3], color, thickness)
        cv2.line(img, bev_box[3], bev_box[0], color, thickness)
    
    # cv2.imwrite(f'{save_dir}/{basename}.png', res)
    # return img

def project_world_to_fisheye_coordinate(world_locations):
    
    locations_tmp = world_locations 
    # print("cam_locations: ",np.shape(cam_locations))
    x_global = (locations_tmp[:,:,0,:]).reshape(-1,)
    y_global = (locations_tmp[:,:,1,:]).reshape(-1,)
    z_global = (locations_tmp[:,:,2,:]).reshape(-1,)

    # z_1 = matR[0][0]*x_global + matR[0][1]*y_global + matR[0][2]*z_global + vecT[0]
    # x_1 = matR[1][0]*x_global + matR[1][1]*y_global + matR[1][2]*z_global + vecT[1]
    # y_1 = matR[2][0]*x_global + matR[2][1]*y_global + matR[2][2]*z_global + vecT[2]

    x_1 = matR[0][0]*x_global + matR[0][1]*y_global + matR[0][2]*z_global + vecT[0]
    y_1 = matR[1][0]*x_global + matR[1][1]*y_global + matR[1][2]*z_global + vecT[1]
    z_1 = matR[2][0]*x_global + matR[2][1]*y_global + matR[2][2]*z_global + vecT[2]


    p_xy_shape = locations_tmp.shape 
    res_pc = torch.zeros_like(locations_tmp)    

    res_pc_tmp = x_1    
    res_pc_tmp = res_pc_tmp.reshape((p_xy_shape[0],p_xy_shape[1],1,p_xy_shape[3]))
    res_pc[:,:,0:1,:] = res_pc_tmp 
    
    res_pc_tmp = y_1
    res_pc_tmp = res_pc_tmp.reshape((p_xy_shape[0],p_xy_shape[1],1,p_xy_shape[3]))
    res_pc[:,:,1:2,:] = res_pc_tmp

    res_pc_tmp = z_1
    res_pc_tmp = res_pc_tmp.reshape((p_xy_shape[0],p_xy_shape[1],1,p_xy_shape[3]))
    res_pc[:,:,2:3,:] = res_pc_tmp

    # rot_y = rot_y.reshape((p_xy_shape[0],p_xy_shape[1],p_xy_shape[3]))

    return res_pc

def transform_fisheye_to_world_coordinate(cam_locations):

        # print("cam_locations: ",np.shape(cam_locations))
        # xc = (cam_locations[:,:,2,:]).reshape(-1,)
        # yc = (cam_locations[:,:,1,:]).reshape(-1,)
        # zc = (cam_locations[:,:,0,:]).reshape(-1,)
        N = cam_locations.shape[0]
 
        # xc = -(cam_locations[:,:,2,:]).reshape(-1,)          #   x -> depth
        # yc = (cam_locations[:,:,0,:]).reshape(-1,)          #   y -> width
        # zc = (cam_locations[:,:,1,:]).reshape(-1,)   - 1.65       #   z -> height   
        # breakpoint()
        xc = -(cam_locations[:,:,2,:]).reshape(1,N) + vecT_svm[0]         #   x -> depth
        yc = (cam_locations[:,:,0,:]).reshape(1,N) + vecT_svm[1]          #   y -> width
        zc = (cam_locations[:,:,1,:]).reshape(1,N) + vecT_svm[2]         #   z -> height   
        # breakpoint()
        res_pc = torch.cat([xc, yc, zc], dim = 0)
        # vecT = [0,0,0]
        

        #   Change coord in the extrinsic matrix 
        # x_1 = matR_inv[1][0]*(xc-vecT[1]) + matR_inv[1][1]*(yc-vecT[2]) + matR_inv[1][2]*(zc-vecT[0])
        # y_1 = matR_inv[2][0]*(xc-vecT[1]) + matR_inv[2][1]*(yc-vecT[2]) + matR_inv[2][2]*(zc-vecT[0])
        # z_1 = matR_inv[0][0]*(xc-vecT[1]) + matR_inv[0][1]*(yc-vecT[2]) + matR_inv[0][2]*(zc-vecT[0])
        # x_1 = matR_inv[0][0]*(xc-vecT[0]) + matR_inv[0][1]*(yc-vecT[1]) + matR_inv[0][2]*(zc-vecT[2])
        # y_1 = matR_inv[1][0]*(xc-vecT[0]) + matR_inv[1][1]*(yc-vecT[1]) + matR_inv[1][2]*(zc-vecT[2])
        # z_1 = matR_inv[2][0]*(xc-vecT[0]) + matR_inv[2][1]*(yc-vecT[1]) + matR_inv[2][2]*(zc-vecT[2])
        # breakpoint()

        # p_xy_shape = cam_locations.shape 
        # res_pc = torch.zeros_like(cam_locations)    

        # res_pc_tmp = xc
        # res_pc_tmp = res_pc_tmp.reshape((p_xy_shape[0],p_xy_shape[1],1,p_xy_shape[3]))
        # res_pc[:,:,0:1,:] = res_pc_tmp 

        # res_pc_tmp = yc
        # res_pc_tmp = res_pc_tmp.reshape((p_xy_shape[0],p_xy_shape[1],1,p_xy_shape[3]))
        # res_pc[:,:,1:2,:] = res_pc_tmp

        # res_pc_tmp = zc
        # res_pc_tmp = res_pc_tmp.reshape((p_xy_shape[0],p_xy_shape[1],1,p_xy_shape[3]))
        # res_pc[:,:,2:3,:] = res_pc_tmp

        return res_pc 

def test_project_to_world_coordinate(abs_corners, extrinsic_matrix):

    tensor_matR = torch.Tensor(matR_inv).cuda() 
    tensor_vecT = torch.Tensor(vecT).cuda()
    abs_corners_world = torch.matmul(abs_corners, tensor_matR.T)
    abs_corners_world[:, :, 0] = abs_corners_world[:, :, 0] + tensor_vecT[0]
    abs_corners_world[:, :, 1] = abs_corners_world[:, :, 1] + tensor_vecT[1]
    abs_corners_world[:, :, 2] = abs_corners_world[:, :, 2] + tensor_vecT[2]

    return abs_corners_world

def transform_perspective_to_fisheye(persp_location):
    x_p, y_p, z_p = persp_location[:, 2], persp_location[:, 0], persp_location[:, 1]
    len3D = torch.sqrt(x_p*x_p + y_p*y_p + z_p*z_p)
    len2D = torch.sqrt(x_p*x_p + y_p*y_p)
    theta = torch.asin(len2D/len3D)
    thetaSq = theta ** theta
    cdist = theta * (1.0 +  matD[0] * thetaSq 
                                + 
                                matD[1] * thetaSq * thetaSq 
                                +
                                matD[2] * thetaSq * thetaSq * thetaSq 
                                +
                                matD[3] * thetaSq * thetaSq * thetaSq * thetaSq
    )
    xd = x_p * cdist/len2D
    yd = y_p * cdist/len2D
    vecPoint2D_u = matK[0][0] * xd + matK[0][2]
    vecPoint2D_v = matK[1][1] * yd + matK[1][2]

    results = torch.zeros_like(persp_location)
    results[:, 0] = vecPoint2D_u
    results[:, 1] = vecPoint2D_v
    breakpoint()
    return result
def in_camera_coordinate(dimension, center, angle, axis: str, is_homogenous=False):
        assert axis in ['rotx', 'roty', 'rotz'], "Not valid axis for rotation"
        """
        dimension: [w, h, l]
        center:    [x, y, z]
        """

        rot_func  = eval(axis) 
        h, w, l = dimension
        # l, h, w = dimension
        x_0, y_0, z_0 = center
        
        angle_offset = np.pi / 2.0
        xPos = 0
        yPos = 0
        zPos = 0
        cornerPoints = np.array([[xPos, yPos, zPos]])
        #   World coordinate 
        x = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
        z = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]

        # x = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        # y = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
        # z = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]

        box_coord = np.stack([x,y,z])
        box_coord = np.hstack([box_coord, cornerPoints.T])
         
        
        # Rot =  rot_func('rotz')
        Rot = rotz(angle - angle_offset)
        points_3d =  Rot @ box_coord

        # cornerPointsLen = points_3d.shape[1]
        #   Convert to cam coord
        # self.t = self.project_center_world2fisheye()
        # Translation
        points_3d[0, :] = points_3d[0, :] + x_0
        points_3d[1, :] = points_3d[1, :] + y_0
        points_3d[2, :] = points_3d[2, :] + z_0

        # breakpoint()
        cam_points_3d = points_3d 
        # cam_points_3d = self.convert_world2fisheye_camback(points_3d) 

        
        if is_homogenous:
            points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

        return cam_points_3d 

def calc_corners_world_coord2(world_center, dimension, rotation):
    pass
########################################################################################################3
def denorm(image, cfg):
    new_image = np.array((image * cfg.data.augmentation.rgb_std +  cfg.data.augmentation.rgb_mean) * 255, dtype=np.uint8)
    return new_image

@PIPELINE_DICT.register_module
@torch.no_grad()
def evaluate_kitti_depth(cfg:EasyDict, 
                       model:nn.Module,
                       dataset_val:Sequence,
                       writer:SummaryWriter,
                       epoch_num:int, 
                       result_path_split='validation'
                       ):
    model.eval()
    result_path = os.path.join(cfg.path.preprocessed_path, result_path_split, 'data')
    if os.path.isdir(result_path):
        os.system("rm -r {}".format(result_path))
        print("clean up the recorder directory of {}".format(result_path))
    os.mkdir(result_path)
    print("rebuild {}".format(result_path))
    for index in tqdm(range(len(dataset_val))):
        data = dataset_val[index]
        collated_data = dataset_val.collate_fn([data])
        image, K = collated_data
        return_dict = model(
                [image.cuda().float(), image.new(K)]
            )
        depth = return_dict["target"][0, 0]
        depth_uint16 = (depth * 256).cpu().numpy().astype(np.uint16)
        w, h = data['original_shape'][1], data['original_shape'][0]
        height_to_pad = h - depth_uint16.shape[0]
        depth_uint16 = np.pad(depth_uint16, [(height_to_pad, 0), (0, 0)], mode='edge')
        depth_uint16 = cv2.resize(depth_uint16, (w, h))
        depth_uint16[depth_uint16 == 0] = 1 
        image_name = "%010d.png" % index
        cv2.imwrite(os.path.join(result_path, image_name), depth_uint16)

    if "is_running_test_set" in cfg and cfg["is_running_test_set"]:
        print("Finish evaluation.")
        return
    result_texts = evaluate_depth(
        label_path = os.path.join(cfg.path.validation_path, 'groundtruth_depth'),
        result_path = result_path
    )
    for index, result_text in enumerate(result_texts):
        if writer is not None:
            writer.add_text("validation result {}".format(index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
        print(result_text, end='')
    print()

@PIPELINE_DICT.register_module
@torch.no_grad()
def evaluate_kitti_obj(cfg:EasyDict, 
                       model:nn.Module,
                       dataset_val:Sized,
                       writer:SummaryWriter,
                       epoch_num:int,
                       result_path_split='validation'
                       ):
    model.eval()
    result_path = os.path.join(cfg.path.preprocessed_path, result_path_split, 'data')
    # breakpoint()
    if os.path.isdir(result_path):
        os.system("rm -r {}".format(result_path))
        print("clean up the recorder directory of {}".format(result_path))
    os.mkdir(result_path)
    print("rebuild {}".format(result_path))

    """
    Get the function testing in here
    """
    test_func = PIPELINE_DICT[cfg.trainer.test_func]
    projector = BBox3dProjector().cuda()
    backprojector = BackProjection().cuda()

    """Annotations path for visualization"""

    annotation_dir = result_path[:-4] + 'annotations'
    result_base    = result_path[:-4] + 'images_base'
    for index in tqdm(range(len(dataset_val))):
        
        test_one(cfg, index, dataset_val, model, test_func, backprojector, projector, result_path, annotation_dir, result_base)
    if "is_running_test_set" in cfg and cfg["is_running_test_set"]:
        print("Finish evaluation.")
        return
    result_texts = evaluate(
        label_path=os.path.join(cfg.path.data_path, 'label_2'),
        result_path=result_path,
        label_split_file=cfg.data.val_split_file,
        current_classes=[i for i in range(len(cfg.obj_types))],
        gpu=min(cfg.trainer.gpu, torch.cuda.device_count() - 1)
    )
    for class_index, result_text in enumerate(result_texts):
        if writer is not None:
            writer.add_text("validation result {}".format(class_index), result_text.replace(' ', '&nbsp;').replace('\n', '  \n'), epoch_num + 1)
        print(result_text)

def test_one(cfg, index, dataset, model, test_func, backprojector:BackProjection, projector:BBox3dProjector, result_path, annotation_path, result_base):
    data = dataset[index]
    if isinstance(data['calib'], list):
        P2 = data['calib'][0]
    else:
        P2 = data['calib']
    original_height = data['original_shape'][0]
    
    # collated_data, collated_data_transform, calib = dataset.collate_fn([data])
    collated_data = dataset.collate_fn([data])
    height = collated_data[0].shape[2]

    
    scores, bbox, obj_names = test_func(collated_data, model, None, cfg=cfg)
    colors = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue':[0, 0, 255]
    }
    
    bbox_2d = bbox[:, 0:4]

    vis_bev = True # visualize bev or not
    if bbox.shape[1] > 4: # run 3D
        bbox_3d_state = bbox[:, 4:] #[cx,cy,z,w,h,l,alpha, bot, top]
        # bbox_3d_state_transform = bbox_transform[:, 4:] #[cx,cy,z,w,h,l,alpha, bot, top]
        
        # bbox_3d_state[:, 6] = bbox_3d_state[:, 6] - np.pi/16

        # bbox_3d_state_3d = backprojector(bbox_3d_state, P2) #[x, y, z, w, h ,l, alpha, bot, top]
        # _, _, thetas = projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))

        # start newly added

        # new intrinsic matrix
        P_temp = np.array([[800, 0, 720, 0],
                           [0, 800, 224, 0],
                           [0, 0, 1, 2.745884e-03]]) # 8.00e+02, #798.245 # 6.4e+02

        # P_ori = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        #                 [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        #                 [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])

        # K = np.float32([[127.125, 0, 638.596],
        #         [0, 127.155, 399.95],
        #         [0, 0, 1]]) # intrinsic of SVM

        P2 = P_temp

        #   abs_corners: [N, 8, 3]
        #   bbox_3d_corner_homo: [N, 8 ,3]
        #   bbox_3d_state_3d: []
        bbox_3d_state_3d = backprojector(bbox_3d_state, P2) #[x, y, z, w,h ,l, alpha, bot, top]          
        # bbox_3d_state_3d_transform = backprojector(bbox_3d_state_transform, P2) #[x, y, z, w,h ,l, alpha, bot, top]          
        
        abs_corners, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, bbox_3d_state_3d.new(P2))
        # abs_corners_transform, bbox_3d_corner_homo_transform, thetas_transform = projector(bbox_3d_state_3d_transform, bbox_3d_state_3d_transform.new(P2))
         
        """
        Project the center from camera perspective to camera fisheye coord
        Project the center from camera fisheye coord to world coordinate
        """
        cam_center = bbox_3d_state_3d[:, :3]
        # cam2fisheye_center = transform_perspective_to_fisheye(cam_center)
        cam2world_center = transform_fisheye_to_world_coordinate(cam_center.view(cam_center.shape[0], 1, 3, 1))

        """Project the transform center from camera coord to world coor"""
        # cam_center_transform = bbox_3d_state_3d_transform[:, :3] 
        # cam2world_center_transform = transform_fisheye_to_world_coordinate(cam_center_transform.view(cam_center.shape[0], 1, 3, 1))
        
        cam2world_center = cam2world_center.squeeze(1).squeeze(-1).T
        # cam2world_center_transform = cam2world_center_transform.squeeze(1).squeeze(-1)
        
        image = collated_data[0]
        img = image.squeeze().permute(1, 2, 0).numpy()
        rgb_image = denorm(img, cfg)
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        
            
        size = (384, 384, 3)
        name = "%06d" % index
        N_corners = abs_corners.shape[0]
        bev_size = (40, 40)     #   [Width, Depth]
        scale_factor = [1.0, 2.0]       #[scale_depth, scale_width]
 
        # canvas_base = cv2.imread(f'{result_base}/{name}.png')
        canvas_base = None
        # canvas_base = canvas_base[:, canvas_base.shape[1]-384:, :]
        if canvas_base is None:
            canvas_base = np.ones(size).astype('uint8') * 255
            canvas_base = cv2.circle(canvas_base, (int(size[1]/2), size[0]-10), 10, (0, 0, 255), -1)

        for idx, (box, corners) in enumerate(zip(bbox_3d_corner_homo, abs_corners)):
            # breakpoint()
            # color = colors[idx]
            # idx+=1
            # color = np.random.randint(0, 235, [1, 3]).tolist()[0]
            box = box.cpu().numpy().T
            cam2world_center = torch.reshape(cam2world_center, (N_corners, 3))
            # cam2world_center_transform = torch.reshape(cam2world_center_transform, (N_corners, 3))

            rgb_image = draw_3D_box(rgb_image, box, colors['red']) 
            # breakpoint()
            # try:
            world_center = [cam2world_center[idx][0].item(), cam2world_center[idx][1].item(), cam2world_center[idx][2].item()]
            # world_center_transform = [cam2world_center_transform[idx][0].item(), cam2world_center_transform[idx][1].item(), cam2world_center_transform[idx][2].item()]
            # except Exception as e :
            #     print(e)
            #     breakpoint()
            # breakpoint()
            # print(f'Center of box {idx + 1} in world coor: {world_center}')
            dimension = [bbox_3d_state_3d[idx][3].item(), bbox_3d_state_3d[idx][4].item(), bbox_3d_state_3d[idx][5].item()]
            # dimension_transform = [bbox_3d_state_3d_transform[idx][3].item(), bbox_3d_state_3d_transform[idx][4].item(), bbox_3d_state_3d_transform[idx][5].item()]
            # world_corners = calc_corners_world_coord(world_center, dimension, thetas[idx]).squeeze(0).squeeze(0) 
            # new_corners = torch.transpose(world_corners, 0, 1)
            
            # breakpoint()
            # world_center = [cam2world_center[idx][2].item(), cam2world_center[idx][0].item(), cam2world_center[idx][1].item()]
            world_corners = in_camera_coordinate(dimension, world_center, thetas[idx].item(), 'rotz', False)
            # world_corners_transform = in_camera_coordinate(dimension_transform, world_center_transform, thetas_transform[idx].item(), 'rotz', False)
 
            # breakpoint()
            

            if vis_bev: 
                
                # bev_box = torch.cat([corners[:2], corners[4:6]])[:, ::2]
                # bev_box = new_corners[:, :2] 
                # bev_box = bev_box.detach().cpu().numpy()
                new_corners   = world_corners.T
                # new_corners_transform = world_corners_transform.T 
                bev_box = new_corners[:, :2]
                # bev_box_transform = new_corners_transform[:, :2]

                # breakpoint()
                
                # breakpoint()
                bev_box[:, 0] = size[1]/scale_factor[0] + bev_box[:, 0] * size[1]/bev_size[0]   #   Depth
                bev_box[:, 1] = size[0]/scale_factor[1] + bev_box[:, 1] * size[0]/bev_size[1]   #   Width                 
                bev_box = bev_box[:, ::-1].astype(np.int32)

                """Prepare for visualizing the bev bov after transformation""" 
                # bev_box_transform[:, 0] = size[1]/scale_factor[0] + bev_box_transform[:, 0] * size[1]/bev_size[0]   #   Depth
                # bev_box_transform[:, 1] = size[0]/scale_factor[1] + bev_box_transform[:, 1] * size[0]/bev_size[1]   #   Width                 
                # bev_box_transform = bev_box_transform[:, ::-1].astype(np.int32)
                
                # bev_box = bev_box.astype(np.int32)
                # breakpoint()
                #   If bev_box is torch.tensor -> use int()
                bev_box = [tuple(x) for x in bev_box.tolist()]
                # bev_box_transform = [tuple(x) for x in bev_box_transform.tolist()]
                
                """Visualizing the initial bev box""" 
                color = colors['red']
                cv2.line(canvas_base, bev_box[0], bev_box[1], colors['red'], 3)
                cv2.line(canvas_base, bev_box[1], bev_box[2], colors['red'], 3)
                cv2.line(canvas_base, bev_box[2], bev_box[3], colors['red'], 3)
                cv2.line(canvas_base, bev_box[3], bev_box[0], colors['red'], 3)
                
                """Visualizing the transformed bev box""" 
                # cv2.line(canvas, bev_box_transform[0], bev_box_transform[1], color_transform, 3)
                # cv2.line(canvas, bev_box_transform[1], bev_box_transform[2], color_transform, 3)
                # cv2.line(canvas, bev_box_transform[2], bev_box_transform[3], color_transform, 3)
                # cv2.line(canvas, bev_box_transform[3], bev_box_transform[0], color_transform, 3)

        move_x = 110
        move_y = -150
        scale = 1.8
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (0, 50) 
        color = (255, 0, 0)
        fontScale = 0.5
        thickness = 1
        canvas_base = cv2.putText(canvas_base, f'X:{move_x}; Y:{move_y}; S:{scale}', org, font, fontScale, color, thickness, cv2.LINE_AA) 
        name = "%06d" % index

        # gt_rect_corners = get_ground_truth(f'{name}.json', annotation_path)
        # draw_gt_bev(canvas_base, gt_rect_corners, bev_size, scale_factor)
        # cv2.imwrite('canvas.png', canvas)
        # breakpoint()
        concat = True
        if concat:
            # concat the canvas to the rgb_image
            rgb_image = cv2.hconcat([rgb_image, canvas_base])
        else:
            rgb_image = canvas_base
        # breakpoint()

        folder_name = 'image_tune_base'

        image_path = result_path[:-4] + folder_name
        if not os.path.isdir(image_path):
            os.mkdir(image_path)
        # print(f'Write results to {image_path}')
        cv2.imwrite(f'{image_path}/{name}.png', cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        # end newly added
        # breakpoint()
        original_P = data['original_P']
        scale_x = original_P[0, 0] / P2[0, 0]
        scale_y = original_P[1, 1] / P2[1, 1]
        
        shift_left = original_P[0, 2] / scale_x - P2[0, 2]
        shift_top  = original_P[1, 2] / scale_y - P2[1, 2]
        bbox_2d[:, 0:4:2] += shift_left
        bbox_2d[:, 1:4:2] += shift_top

        bbox_2d[:, 0:4:2] *= scale_x
        bbox_2d[:, 1:4:2] *= scale_y

        write_result_to_file(result_path, index, scores, bbox_2d, bbox_3d_state_3d, thetas, obj_names)
    else:
        if "crop_top" in cfg.data.augmentation and cfg.data.augmentation.crop_top is not None:
            crop_top = cfg.data.augmentation.crop_top
        elif "crop_top_height" in cfg.data.augmentation and cfg.data.augmentation.crop_top_height is not None:
            if cfg.data.augmentation.crop_top_height >= original_height:
                crop_top = 0
            else:
                crop_top = original_height - cfg.data.augmentation.crop_top_height

        scale_2d = (original_height - crop_top) / height
        bbox_2d[:, 0:4] *= scale_2d
        bbox_2d[:, 1:4:2] += cfg.data.augmentation.crop_top
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        write_result_to_file(result_path, index, scores, bbox_2d, obj_types=obj_names)


# def calc_corners_world_coord(world_center, dimension, rotation):
#     """
#     Calculate 8 corners with the transformed center from 
#     cam coordinate to world coordinate
#     """
#     # xPos = world_center[0]
#     xPos = 0
#     yPos = 0
#     zPos = 0
#     # yPos = world_center[1]
#     # zPos = world_center[2]

#     #   dimension = [w, h, l]
#     width, height, length = dimension
    
#     rotation = rotation.detach().cpu().numpy() 
#     _cos = np.cos(rotation)
#     _sin = np.sin(rotation)
#     rot_matrix = torch.tensor(rotz(rotation)).to(torch.float)
#     # rot_matrix = torch.tensor(roty(rotation)).to(torch.float)
    
#     cornerPoints=[]
#     # cornerPoints.append(np.array([xPos - width / 2, yPos -  height/ 2, zPos - length / 2])) #4
#     # cornerPoints.append(np.array([xPos + width / 2, yPos -  height/ 2, zPos - length / 2])) #5
#     # cornerPoints.append(np.array([xPos + width / 2, yPos +  height/ 2, zPos - length / 2])) #6  
#     # cornerPoints.append(np.array([xPos + width / 2, yPos +  height/ 2, zPos + length / 2])) #7

#     # cornerPoints.append(np.array([xPos + width / 2, yPos -  height/ 2, zPos + length / 2])) #4
#     # cornerPoints.append(np.array([xPos - width / 2, yPos -  height/ 2, zPos + length / 2])) #5
#     # cornerPoints.append(np.array([xPos - width / 2, yPos +  height/ 2, zPos + length / 2])) #6  
#     # cornerPoints.append(np.array([xPos - width / 2, yPos +  height/ 2, zPos - length / 2])) #7
    
#     cornerPoints.append(np.array([xPos + length / 2, yPos -  width/ 2, zPos - height / 2])) #4
#     cornerPoints.append(np.array([xPos + length / 2, yPos +  width/ 2, zPos - height / 2])) #5
#     cornerPoints.append(np.array([xPos - length / 2, yPos +  width/ 2, zPos - height / 2])) #6  
#     cornerPoints.append(np.array([xPos - length / 2, yPos -  width/ 2, zPos - height / 2])) #7

#     cornerPoints.append(np.array([xPos + length / 2, yPos -  width/ 2, zPos + height / 2])) #0
#     cornerPoints.append(np.array([xPos + length / 2, yPos +  width/ 2, zPos + height / 2])) #1
#     cornerPoints.append(np.array([xPos - length / 2, yPos +  width/ 2, zPos + height / 2])) #2
#     cornerPoints.append(np.array([xPos - length / 2, yPos -  width/ 2, zPos + height / 2])) #3



#     cornerPointsLen = len(cornerPoints)
#     cornerPoints = torch.tensor(cornerPoints).to(torch.float)     #   [N, 3, 8]
#     # rotated_corners_x = cornerPoints[:, 2] * _cos + cornerPoints[ :, 0] * _sin
#     # rotated_corners_z = cornerPoints[:, 0] * _cos - cornerPoints[:, 2] * _sin
#     # cornerPoints.append(np.array([xPos, yPos, zPos])) #8 center    

#     # cornerPoints[:,0] = rotated_corners_x
#     # cornerPoints[:, 2] = rotated_corners_z

#     # cornerPoints[:, 0] = cornerPoints[:, 0] + world_center[0]
#     # cornerPoints[:, 1] = cornerPoints[:, 1] + world_center[1]
#     # cornerPoints[:, 2] = cornerPoints[:, 2] + world_center[2]
    
#     cornerPoints = cornerPoints.T           #   Reshape to [N, 3, 8] 
#     breakpoint() 
#     ref_rotated_corner_points = torch.matmul(rot_matrix,cornerPoints)  

#     rotated_corner_points = torch.zeros_like(ref_rotated_corner_points)

#     rotated_corner_points[0,:] = ref_rotated_corner_points[0,:]+world_center[0]
#     rotated_corner_points[1,:] = ref_rotated_corner_points[1,:]+world_center[1]
#     rotated_corner_points[2,:] = ref_rotated_corner_points[2,:]+world_center[2] 
#     # breakpoint() 
    
#     batch_points = rotated_corner_points.view(1,1,3,cornerPointsLen)
#     # batch_points = cornerPoints.view(1,1,3,cornerPointsLen)
#     return batch_points

#     # return rotated_corner_points
