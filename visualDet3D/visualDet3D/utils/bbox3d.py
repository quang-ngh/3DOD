import cv2
# import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
def rotx(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def alpha2theta_3d(alpha, x, z, P2):
    """ Convert alpha to theta with 3D position
    Args:
        alpha [torch.Tensor/ float or np.ndarray]: size: [...]
        x     []: size: [...]
        z     []: size: [...]
        P2    [torch.Tensor/ np.ndarray]: size: [3, 4]
    Returns:
        theta []: size: [...]
    """
    # offset = P2[0, 3] / P2[0, 0]
    if isinstance(alpha, torch.Tensor):
        # theta = alpha + torch.atan2(x + offset, z)
        theta = alpha + torch.atan2(x, z)
    else:
        # theta = alpha + np.arctan2(x + offset, z)
        theta = alpha + torch.atan2(x, z)
    return theta

class Box3D:

    def __init__(self, info: dict, true_z: bool):

        box3d_info = info['box3d']
        dimension = box3d_info['dimension']
        location  = box3d_info['location']
        orientation = box3d_info['orientation']
        self.id = int(info['id'])
        self.w = dimension['width']
        self.h = dimension['height']
        self.l = dimension['length']    
        self.x = location['x']
        self.y = location['y']
        self.z = self.h / 2.0
        if true_z:
            self.z = location['z']
        
        self.ry = orientation['rotationYaw'] + (np.pi / 2.0)
        # self.ry = -np.pi / 2.0
        # self.ry = 0
        self.t = [self.x, self.y, self.z]
        self.matR = np.array([
                [-0.015964651480317116,          0.99987155199050903,            0.0014207595959305763],
                [0.58267718553543091,          0.010458142496645451,            -0.81263637542724609],
                [-0.81254684925079346,            -0.012145611457526684,            -0.58276933431625366]
                ])
        self.vecT = np.array([ -0.037088312208652496, 1.5999122858047485, -1.2545629739761353])

    def project_center_world2fisheye(self):

        locations_tmp = np.array(self.t)
        x_global = locations_tmp[0].reshape(-1,)
        y_global = locations_tmp[1].reshape(-1,)
        z_global = locations_tmp[2].reshape(-1,)

        x_1 = self.matR[0][0]*x_global + self.matR[0][1]*y_global + self.matR[0][2]*z_global + self.vecT[0]
        y_1 = self.matR[1][0]*x_global + self.matR[1][1]*y_global + self.matR[1][2]*z_global + self.vecT[1]
        z_1 = self.matR[2][0]*x_global + self.matR[2][1]*y_global + self.matR[2][2]*z_global + self.vecT[2]

        return [x_1[0], y_1[0], z_1[0]]

    def project_world_to_fisheye_coordinate(self,world_locations):
        if not torch.is_tensor(world_locations) :
            world_locations = torch.tensor(world_locations)
        locations_tmp = world_locations 
        # print("cam_locations: ",np.shape(cam_locations))
        x_global = (locations_tmp[:,:,0,:]).reshape(-1,)
        y_global = (locations_tmp[:,:,1,:]).reshape(-1,)
        z_global = (locations_tmp[:,:,2,:]).reshape(-1,)

        x_1 = self.matR[0][0]*x_global + self.matR[0][1]*y_global + self.matR[0][2]*z_global + self.vecT[0]
        y_1 = self.matR[1][0]*x_global + self.matR[1][1]*y_global + self.matR[1][2]*z_global + self.vecT[1]
        z_1 = self.matR[2][0]*x_global + self.matR[2][1]*y_global + self.matR[2][2]*z_global + self.vecT[2]

 

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

    def convert_world2fisheye_camback(self, points_3d):
        cornerPointsLen = points_3d.shape[-1]
        points_3d = np.reshape(points_3d, (1,1,3, cornerPointsLen)) 
        cam_rotated_corner_points = self.project_world_to_fisheye_coordinate(points_3d)
        # breakpoint() 
        cam_points_3d = cam_rotated_corner_points.view(3, cornerPointsLen).detach().cpu().numpy()
        return cam_points_3d

    def in_camera_coordinate(self, axis: str, is_homogenous=False):
        assert axis in ['rotx', 'roty', 'rotz'], "Not valid axis for rotation"

        rot_func  = eval(axis) 
        l = self.l
        w = self.w
        h = self.h

        xPos = 0
        yPos = 0
        zPos = 0
        cornerPoints = np.array([[xPos, yPos, zPos]])
        #   World coordinate 
        x = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
        z = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]

        box_coord = np.stack([x,y,z])
        box_coord = np.hstack([box_coord, cornerPoints.T])
         
        # breakpoint()
        Rot =  rot_func(self.ry)
        points_3d = Rot @ box_coord

        cornerPointsLen = points_3d.shape[1]
        #   Convert to cam coord
        # self.t = self.project_center_world2fisheye()
        # Translation
        points_3d[0, :] = points_3d[0, :] + self.t[0]
        points_3d[1, :] = points_3d[1, :] + self.t[1]
        points_3d[2, :] = points_3d[2, :] + self.t[2]
        
        # points_3d = self.convert_world2fisheye_camback(points_3d) 
        # if is_homogenous:
        #     points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

        # return cam_points_3d 
        return points_3d
 
    def get2DBox(self):
        #   Extract 3D Box information
        yaw = self.rotYaw
        width = self.w
        length = self.l
        x = self.x
        y = self.y 

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Calculate the half-dimensions of the bounding box
        half_width = width / 2.0
        half_length = length / 2.0

        # Calculate the 2D coordinates of the four corners of the bounding box
        x1 = x - half_length * cos_yaw - half_width * sin_yaw
        y1 = y - half_length * sin_yaw + half_width * cos_yaw

        x2 = x - half_length * cos_yaw + half_width * sin_yaw
        y2 = y - half_length * sin_yaw - half_width * cos_yaw

        x3 = x + half_length * cos_yaw + half_width * sin_yaw
        y3 = y + half_length * sin_yaw - half_width * cos_yaw

        x4 = x + half_length * cos_yaw - half_width * sin_yaw
        y4 = y + half_length * sin_yaw + half_width * cos_yaw
    
    def __str__(self):
        return f'Dimension (W-H-L): {self.w} - {self.h} - {self.l}\nLocation (x-y-z): {self.x} - {self.y} - {self.z}\nRotation: {self.ry}'

class BBox3dProjector(nn.Module):
    """
        forward methods
            input:
                unnormalize bbox_3d [N, 7] with  x, y, z, w, h, l, alpha
                tensor_p2: tensor of [3, 4]
            output:
                [N, 8, 3] with corner point in camera frame
                [N, 8, 3] with corner point in image frame
                [N, ] thetas
    """
    def __init__(self):
        super(BBox3dProjector, self).__init__()
        self.register_buffer('corner_matrix', torch.tensor(
            [[-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [ 1,  1,  1],
            [ 1, -1,  1],
            [-1, -1,  1],
            [-1,  1,  1],
            [-1,  1, -1]]
        ).float()  )# 8, 3

    def forward(self, bbox_3d, tensor_p2):
        """
            input:
                unnormalize bbox_3d [N, 7] with  x, y, z, w, h, l, alpha
                tensor_p2: tensor of [3, 4]
            output:
                [N, 8, 3] with corner point in camera frame # 8 is determined by the shape of self.corner_matrix
                [N, 8, 3] with corner point in image frame
                [N, ] thetas
        """
        tensor_p2 = tensor_p2.to(DEVICE)
        relative_eight_corners = 0.5 * self.corner_matrix * bbox_3d[:, 3:6].unsqueeze(1)  # [N, 8, 3]
        # [batch, N, ]
        # alpha = bbox_3d[:, -1] - torch.atan2(bbox_3d[:,0], bbox_3d[:,2])
        # thetas = alpha2theta_3d(alpha, bbox_3d[..., 0], bbox_3d[..., 2], tensor_p2)
        thetas = bbox_3d[:, -1]
        _cos = torch.cos(thetas).unsqueeze(1)  # [N, 1]
        _sin = torch.sin(thetas).unsqueeze(1)  # [N, 1]
        rotated_corners_x, rotated_corners_z = (
            relative_eight_corners[:, :, 2] * _cos +
                relative_eight_corners[:, :, 0] * _sin,
        -relative_eight_corners[:, :, 2] * _sin +
            relative_eight_corners[:, :, 0] * _cos
        )  # relative_eight_corners == [N, 8, 3]
        rotated_corners = torch.stack([rotated_corners_x, relative_eight_corners[:,:,1], rotated_corners_z], dim=-1) #[N, 8, 3]
        abs_corners = rotated_corners + \
            bbox_3d[:, 0:3].unsqueeze(1)  # [N, 8, 3]

        # camera_corners = torch.cat([abs_corners,
        #     abs_corners.new_ones([abs_corners.shape[0], self.corner_matrix.shape[0], 1])],
        #     dim=-1).unsqueeze(3)  # [N, 8, 4, 1]
        camera_corners = abs_corners.unsqueeze(-1)
        # breakpoint()
        camera_coord = torch.matmul(tensor_p2, camera_corners).squeeze(-1)  # [N, 8, 3]

        homo_coord = camera_coord / (camera_coord[:, :, 2:] + 1e-6) # [N, 8, 3]

        return abs_corners, homo_coord, thetas

class BackProjection(nn.Module):
    """
        forward method:
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
    """
    def forward(self, bbox3d, p2):
        """
            bbox3d: [N, 7] homo_x, homo_y, z, w, h, l, alpha
            p2: [3, 4]
            return [x3d, y3d, z, w, h, l, alpha]
        """
        fx = p2[0, 0]
        fy = p2[1, 1]
        cx = p2[0, 2]
        cy = p2[1, 2]
        tx = p2[0, 3]
        ty = p2[1, 3]

        z3d = bbox3d[:, 2:3] #[N, 1]
        x3d = (bbox3d[:,0:1] * z3d - cx * z3d - tx) / fx #[N, 1]
        y3d = (bbox3d[:,1:2] * z3d - cy * z3d - ty) / fy #[N, 1]
        return torch.cat([x3d, y3d, bbox3d[:, 2:]], dim=1)