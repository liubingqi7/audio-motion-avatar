import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

from omegaconf import DictConfig

class SMPLXDecoder(nn.Module):
    """
    SMPLX模型参数维度说明：

    Speech数据集/IDOL数据集：
      
    身体部分 (Body):
    - body_joint_num: 22 (包括根关节)
    - body_root_pose: [batch_size, 6] (根关节姿势，使用6D表示法)
    - body_pose: [batch_size, (body_joint_num-1)*6] (身体姿势，不包括根关节)
    - body_shape: [batch_size, 10] (身体形状参数，即beta参数)
    
    手部 (Hands):
    - hand_joint_num: 15 (每只手的关节数)
    - hand_root_pose: [batch_size, 2*6] (左右手根关节姿势)
    - hand_pose: [batch_size, 2*hand_joint_num*6] (左右手姿势)
    
    面部 (Face):
    - face_root_pose: [batch_size, 6] (面部根关节姿势)
    - face_expression: [batch_size, 10] (面部表情参数)
    - face_jaw_pose: [batch_size, 6] (下颌姿势)

    相机参数:
    - focal: [batch_size, 2] (焦距)
    
    Transformer输入输出:
    - smpl_token_dim: 每个token的特征维度
    - smpl_token_len: token序列长度
    - tokens: [batch_size, smpl_token_len, smpl_token_dim]
    - mlp输入: [batch_size, smpl_token_dim * smpl_token_len]
    - 特征向量: [batch_size, 256]
    """
    def __init__(self, cfg: DictConfig = None):
        super(SMPLXDecoder, self).__init__()
        self.cfg = cfg
        self.smpl_token_dim = cfg.smpl_token_dim
        self.smpl_token_len = cfg.smpl_token_len
        
        self.mlp = nn.Sequential(
            nn.Linear(self.smpl_token_dim * self.smpl_token_len, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Get SMPLX model joint counts and parameter dimensions
        self.body_joint_num = 22  # Body joint count (including root joint)
        self.hand_joint_num = 15  # Joint count for each hand
        self.shape_dim = 10       # Shape parameter dimension
        self.expression_dim = cfg.num_expression_coeffs  # Expression parameter dimension
        
        # Decoder heads
        # Body parameters
        self.dec_body_root_pose = nn.Linear(256, 6)  # Root joint pose (6D representation)
        self.dec_body_pose = nn.Linear(256, (self.body_joint_num-1)*6)  # Body pose (6D representation)
        self.dec_body_shape = nn.Linear(256, self.shape_dim)  # Body shape
        self.dec_transl = nn.Linear(256, 3)  # Translation
        
        # Camera parameters
        # self.dec_focal = nn.Linear(256, 2)  # Focal length
        # self.dec_princpt = nn.Linear(256, 2)  # Principal point
        
        # Hand parameters
        # self.dec_hand_root_pose = nn.Linear(256, 2*6)  # Left and right hand root joint poses (6D representation)
        self.dec_hand_pose = nn.Linear(256, 2*self.hand_joint_num*6)  # Left and right hand poses (6D representation)
        
        # Face parameters
        # self.dec_face_root_pose = nn.Linear(256, 6)  # Face root joint pose (6D representation)
        self.dec_face_expression = nn.Linear(256, self.expression_dim)  # Face expression
        self.dec_face_jaw_pose = nn.Linear(256, 6)  # Jaw pose (6D representation)
        self.dec_leye_pose = nn.Linear(256, 6)  # Left eye pose (6D representation)
        self.dec_reye_pose = nn.Linear(256, 6)  # Right eye pose (6D representation)
        
    def forward(self, tokens):
        batch_size = tokens.shape[0]
        
        x = tokens.reshape(batch_size, -1)
        features = self.mlp(x)
        
        pred_body_root_pose_6d = self.dec_body_root_pose(features)
        pred_body_pose_6d = self.dec_body_pose(features)
        pred_body_betas = self.dec_body_shape(features)
        pred_transl = self.dec_transl(features)
        
        # pred_focal = self.dec_focal(features)
        # pred_princpt = self.dec_princpt(features)
        
        # pred_hand_root_pose_6d = self.dec_hand_root_pose(features)
        pred_hand_pose_6d = self.dec_hand_pose(features)
        
        # pred_face_root_pose_6d = self.dec_face_root_pose(features)
        pred_face_expression = self.dec_face_expression(features)
        pred_face_jaw_pose_6d = self.dec_face_jaw_pose(features)
        pred_leye_pose_6d = self.dec_leye_pose(features)
        pred_reye_pose_6d = self.dec_reye_pose(features)
        
        pred_global_orient = matrix_to_axis_angle(rotation_6d_to_matrix(pred_body_root_pose_6d)).reshape(batch_size, 1, 3)
        
        body_pose_matrix = rotation_6d_to_matrix(pred_body_pose_6d.reshape(batch_size, 21, 6))
        pred_body_pose = matrix_to_axis_angle(body_pose_matrix).reshape(batch_size, 21, 3)
        
        # lhand_root_pose_matrix = rotation_6d_to_matrix(pred_hand_root_pose_6d[:, :6].reshape(batch_size, 1, 6))
        # rhand_root_pose_matrix = rotation_6d_to_matrix(pred_hand_root_pose_6d[:, 6:].reshape(batch_size, 1, 6))
        
        lhand_pose_matrix = rotation_6d_to_matrix(pred_hand_pose_6d[:, :self.hand_joint_num*6].reshape(batch_size, self.hand_joint_num, 6))
        rhand_pose_matrix = rotation_6d_to_matrix(pred_hand_pose_6d[:, self.hand_joint_num*6:].reshape(batch_size, self.hand_joint_num, 6))
        
        pred_left_hand_pose = matrix_to_axis_angle(lhand_pose_matrix).reshape(batch_size, self.hand_joint_num, 3)
        pred_right_hand_pose = matrix_to_axis_angle(rhand_pose_matrix).reshape(batch_size, self.hand_joint_num, 3)
        
        # face_root_pose_matrix = rotation_6d_to_matrix(pred_face_root_pose_6d.reshape(batch_size, 1, 6))
        jaw_pose_matrix = rotation_6d_to_matrix(pred_face_jaw_pose_6d.reshape(batch_size, 1, 6))
        leye_pose_matrix = rotation_6d_to_matrix(pred_leye_pose_6d.reshape(batch_size, 1, 6))
        reye_pose_matrix = rotation_6d_to_matrix(pred_reye_pose_6d.reshape(batch_size, 1, 6))
        
        pred_jaw_pose = matrix_to_axis_angle(jaw_pose_matrix).reshape(batch_size, 3)
        pred_leye_pose = matrix_to_axis_angle(leye_pose_matrix).reshape(batch_size, 3)
        pred_reye_pose = matrix_to_axis_angle(reye_pose_matrix).reshape(batch_size, 3)
        
        pred_params = {
            'betas': pred_body_betas,
            'transl': pred_transl,
            'global_orient': pred_global_orient.squeeze(1),
            'body_pose': pred_body_pose,
            'left_hand_pose': pred_left_hand_pose,
            'right_hand_pose': pred_right_hand_pose,
            'jaw_pose': pred_jaw_pose,
            'leye_pose': pred_leye_pose,
            'reye_pose': pred_reye_pose,
            # 'pad_ratio': pred_pad_ratio.squeeze(-1),
            # 'focal': pred_focal,
            # 'princpt': pred_princpt,
            'expression': pred_face_expression
        }
        
        return pred_params
