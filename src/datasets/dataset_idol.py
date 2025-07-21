import os
import random
import numpy as np
import torch
import json
import pickle
import sys
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import pickle
from torch.utils.data import Dataset
# import webdataset as wds
# from lib.utils.train_util import print

import cv2
import av

from omegaconf import OmegaConf, ListConfig

def load_pose(path):
    with open(path, 'rb') as f:
        pose_param = json.load(f)
    c2w = np.array(pose_param['cam_param'], dtype=np.float32).reshape(4,4)
    cam_center = c2w[:3, 3]
    w2c = np.linalg.inv(c2w)
    # pose[:,:2] *= -1
    # pose = np.loadtxt(path, dtype=np.float32, delimiter=' ').reshape(9, 4)
    return [torch.from_numpy(w2c), torch.from_numpy(cam_center)]

def load_npy(file_path):
    return np.load(file_path, allow_pickle=True)

def load_smpl(path, smpl_type='smpl'):
    filetype = path.split('.')[-1]
    with open(path, 'rb') as f:
        if filetype=='pkl':
            smpl_param_data = pickle.load(f)
        elif filetype == 'json':
            smpl_param_data = json.load(f)
        else:
            assert False

    if smpl_type=='smpl':
        with open(os.path.join(os.path.split(path)[0][:-5], 'pose', '000_000.json'), 'rb') as f:
            tf_param = json.load(f)
        smpl_param = np.concatenate([np.array(tf_param['scale']).reshape(1, -1), np.array(tf_param['center'])[None], 
                    smpl_param_data['global_orient'], smpl_param_data['body_pose'].reshape(1, -1), smpl_param_data['betas']], axis=1)
    elif smpl_type == 'smplx':

        tf_param = np.load(os.path.join(os.path.dirname(os.path.dirname(path)), 'scale_offset.npy'), allow_pickle=True).item()
        # smpl_param = np.concatenate([np.array([tf_param['scale']]).reshape(1, -1), tf_param['offset'].reshape(1, -1), 
        smpl_param = np.concatenate([np.array([[1]]), np.array([[0,0,0]]), 
                    np.array(smpl_param_data['global_orient']).reshape(1, -1), np.array(smpl_param_data['body_pose']).reshape(1, -1), 
                    np.array(smpl_param_data['betas']).reshape(1, -1), np.array(smpl_param_data['left_hand_pose']).reshape(1, -1), np.array(smpl_param_data['right_hand_pose']).reshape(1, -1),
                    np.array(smpl_param_data['jaw_pose']).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1), 
                    np.array(smpl_param_data['expression']).reshape(1, -1)], axis=1)
    else:
        assert False
    
    return torch.from_numpy(smpl_param.astype(np.float32)).reshape(-1)


class AvatarDataset(Dataset):
    def __init__(self,
                 data_prefix,
                 code_dir=None,
                #  code_only=False,
                 load_imgs=True,
                 load_norm=False,
                 specific_observation_idcs=None,
                 specific_observation_num=None,
                 first_is_front=False, # yy add  # If True, it will returns a random sampled batch with the front view in the first place
                 better_range=False, # yy add  # If True, the views will not be fully random, but will be selected by a better skip
                 if_include_video_ref_img= False,# yy add Define a variable to indicate whether to include reference images from the video
                 prob_include_video_ref_img= 0.2, # yy add Define a variable to specify the probability
                 allow_k_angles_near_the_front = 0, # yy add, if value > 0, the front view will be allowed to be selected from the range of [front_view - allow_k_angles_near_the_front, front_view + allow_k_angles_near_the_front]
                #  num_test_imgs=0, 
                 if_use_swap_face_v1=False, # yy add, if True, use the swap face v1
                 random_test_imgs=False,
                 scene_id_as_name=False,
                 cache_path=None,
                 cache_repeat=None, # be the same length with the cache_path
                 test_pose_override=None,
                 num_train_imgs=-1,
                 load_cond_data=True,
                 load_test_data=True, 
                 max_num_scenes=-1,  # for debug or testing
                #  radius=0.5,
                 radius=1.0,
                 img_res=[640, 896],
                 test_mode=False,
                 step=1,  # only for debug & visualization purpose
                 crop=False # randomly crop the image with upper body inputs
                 ):
        super(AvatarDataset, self).__init__()
        self.data_prefix = data_prefix
        self.code_dir = code_dir
        # self.code_only = code_only
        self.load_imgs = load_imgs
        self.load_norm = load_norm
        self.specific_observation_idcs = specific_observation_idcs
        self.specific_observation_num = specific_observation_num
        self.first_is_front = first_is_front
        


        self.if_include_video_ref_img= if_include_video_ref_img 
        self.prob_include_video_ref_img = prob_include_video_ref_img
        self.allow_k_angles_near_the_front = allow_k_angles_near_the_front 

        self.better_range = better_range
        # self.num_test_imgs = num_test_imgs
        self.random_test_imgs = random_test_imgs
        self.scene_id_as_name = scene_id_as_name
        self.cache_path = cache_path
        self.cache_repeat = cache_repeat
        self.test_pose_override = test_pose_override
        self.num_train_imgs = num_train_imgs
        self.load_cond_data = load_cond_data
        self.load_test_data = load_test_data
        self.max_num_scenes = max_num_scenes
        self.step = step

        self.if_use_swap_face_v1 = if_use_swap_face_v1
        # import ipdb; ipdb.set_trace()

        self.img_res = [int(i) for i in img_res]

        self.radius = torch.tensor([radius], dtype=torch.float32).expand(3)
        self.center = torch.zeros_like(self.radius)

        self.load_scenes()
        
        self.crop = crop


        self.test_poses = self.test_intrinsics = None

        self.defalut_focal = 1120 #40 * (self.img_res[0]/32) # focal 80mm, sensor 32mm

        self.default_fxy_cxy = torch.tensor([self.defalut_focal, self.defalut_focal,  self.img_res[1]//2, self.img_res[0]//2]).reshape(1, 4)

        self.test_mode = test_mode

        if self.test_mode:
            self.parse_scene = self.parse_scene_test


    def load_scenes(self):

        
        if  isinstance(self.cache_path, ListConfig):

            cache_list = []
            case_num_per_dataset = 1000000000
            for ii, path in enumerate(self.cache_path):
                cache = np.load(path, allow_pickle=True)
                if self.cache_repeat is not None:
                    cache = np.repeat(cache, self.cache_repeat[ii], axis=0)
                print("done loading ", path)
                cache_list.extend(cache[:case_num_per_dataset]) 
            scenes = cache_list
            print(f"=========intialized totally {len(scenes)} scenes===========")

        else:
            if self.cache_path is not None and os.path.exists(self.cache_path):
                scenes = np.load(self.cache_path, allow_pickle=True)
                print("load ", self.cache_path)
            else:
                print(f"{self.cache_path} is not exist")
                raise  FileNotFoundError(f"maybe {self.cache_path} is not exist")
            
        end = len(scenes)
        if self.max_num_scenes >= 0:
            end = min(end, self.max_num_scenes * self.step)
        
        self.scenes = scenes[:end:self.step]
        self.num_scenes = len(self.scenes)
        
    def parse_scene(self, scene_id):
        scene = self.scenes[scene_id]
        input_is_video = False # flag of if the input is video, some operations should be different
        # print(scene)
        # scene['video_path'] = "/data/jxlv/transformers/src/A_pose_MEN-Denim-id_00000089-01_7_additional/result.mp4"
        # scene['image_paths'] = None #"/data/jxlv/transformers/src/A_pose_MEN-Denim-id_00000089-01_7_additional/source_seg.png"
        # import pdb
        # pdb.set_trace()
        # =========== loading the params ===========
        param = np.load(scene['param_path'], allow_pickle=True).item()
        scene.update(param)
        # print(scene.keys())

        # =========== loading the multi-view images ===========
        if scene['image_paths'] is None:
            input_is_video = True
            video_path = scene['video_path']
            try:
                if self.if_use_swap_face_v1:
                    image_paths_or_video = read_frames(scene['video_path'].replace('result.mp4', 'output.mp4'))
                else:
                    image_paths_or_video = read_frames(scene['video_path'])
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error in reading the video : {scene['video_path'].replace('result.mp4', 'output.mp4')}")
                image_paths_or_video = read_frames(scene['video_path'])
            # if 'pose_animate_service_0727' in video_path or 'flux' in video_path:
            #     # move the first to the last
            #     # TODO fixed this bug with a better cameras parameters
            #     image_paths_or_video = image_paths_or_video[1:] + image_paths_or_video[0:1]
        if not input_is_video:
            image_paths_or_video = scene['image_paths']
        scene_name = f"{scene_id:0>4d}" #  image_paths[0].split('/')[-3]
        results = dict(
            scene_id=[scene_id],
            scene_name=
                '{:04d}'.format(scene_id) if self.scene_id_as_name else scene_name,
                # cpu_only=True
                )
        # import pdb; pdb.set_trace()
        # if not self.code_only:
        poses = scene['poses']
        smpl_params = scene['smpl_params']

        # if input_is_video:
        #     num_imgs = len(video)
        # else:
        num_imgs = len(image_paths_or_video)
        # front_view = num_imgs // 4
        # randonly / specificically select the views of output
        smplx_cam_rotate = smpl_params[4: 7] #get global orient # 1, 3, 63, 10
        # smpl_params[70:80] = torch.rand_like(smpl_params[70:80]); print("error !! need to delete this rand betas in avatarnet:287") # get betas
        front_view = find_front_camera_by_rotation(poses, smplx_cam_rotate) # inputs camera poses and smplx poses
        if self.allow_k_angles_near_the_front > 0:
            allow_n_views_near_the_front =  round(self.allow_k_angles_near_the_front / 360 * num_imgs)
            new_front_view = random.randint(-allow_n_views_near_the_front, allow_n_views_near_the_front) + front_view
            if new_front_view >= num_imgs:
                new_front_view = new_front_view - num_imgs
            elif new_front_view < 0:
                new_front_view = new_front_view + num_imgs
            front_view = new_front_view
            # print("changes the front views ranges", front_view, "+-", allow_n_views_near_the_front)

        if self.specific_observation_idcs is None:  ######### if not specify views ########
            # if self.num_train_imgs >= 0:
            #     num_train_imgs = self.num_train_imgs
            # else:
            num_train_imgs = num_imgs
            if self.random_test_imgs: ###### randomly selected images with self.num_train_imgs ######
                cond_inds = random.sample(range(num_imgs), self.num_train_imgs)
            elif self.specific_observation_num: ###### randomly selected "specific_observation_num" images ######
                if self.first_is_front and self.specific_observation_num < 2:
                    # self.specific_observation_num = 2
                    cond_inds =torch.tensor([front_view, front_view]) # first for input, second for supervised
                elif self.better_range: # select views by a uniform distribution range
                    if self.first_is_front: # must include the front view
                        num_train_imgs = self.specific_observation_num - 2
                    else:
                        num_train_imgs = self.specific_observation_num
                    skip_range = num_imgs//num_train_imgs
                    # select random views from each range of [skip_range] seperate from [0, skip_range, 2*skip_range, ...], 
                    cond_inds = torch.randperm(num_train_imgs) * skip_range \
                            + torch.randint(low=0, high=skip_range, size=[num_train_imgs])
                    if self.first_is_front: # concat [the first view * 2] to the front of cond_inds
                        cond_inds = torch.cat([torch.tensor([front_view, front_view]), cond_inds])
                        
                else: # previous version, random views are sampled
                    cond_inds = torch.randperm(num_imgs)[:self.specific_observation_num]
            else:
                cond_inds = np.round(np.linspace(0, num_imgs - 1, num_train_imgs)).astype(np.int64)
        else:   ######### selected target views ########
            cond_inds = self.specific_observation_idcs


        test_inds = list(range(num_imgs))

        if self.specific_observation_num: # yy note: if specific_observation_num is not None, then remove the test_inds
            test_inds = []
        else:
            for cond_ind in cond_inds:
                test_inds.remove(cond_ind)
        cond_smpl_param_ref = torch.zeros([189])
        if_use_smpl_param_ref = torch.Tensor([1]) # 默认使用ref smpl, 
        if self.load_cond_data and len(cond_inds) > 0:
            # cond_imgs, cond_poses, cond_intrinsics, cond_img_paths, cond_smpl_param, cond_norm = gather_imgs(cond_inds)
            cond_imgs, cond_poses, cond_intrinsics, cond_img_paths, cond_smpl_param, cond_norm = \
                gather_imgs(cond_inds, poses, image_paths_or_video, smpl_params, 
                           load_imgs=self.load_imgs, load_norm=self.load_norm, 
                           center=self.center, radius=self.radius,
                           input_is_video=input_is_video)
            cond_smpl_param_ref = cond_smpl_param.clone() # the smpl_param_ref for the reference images
            if cond_intrinsics.shape[-1] == 3: # the old data format, which contains the value of camera center instead of fxfycxcy
                cond_intrinsics = self.default_fxy_cxy.clone().repeat(cond_intrinsics.shape[0], 1)
            # import pdb; pdb.set_trace()
            # print("video_path", video_path)
            if input_is_video:
                cond_img_paths = [f"{video_path[:-4]}_{i:0>4d}.png" for i in  range(self.specific_observation_num)] # Replace the .mp4 into index.png
            
            
            if self.if_include_video_ref_img and input_is_video:
                print("if_include_video_ref_img", self.if_include_video_ref_img)
                # 设置一个随机数，如果小于某个概率，那么替换第一张图为另一个图片
                if np.random.rand() < self.prob_include_video_ref_img:
                    if 'ref' in scene:
                        ref_image_path = scene['image_ref']
                        print("ref_image_path",ref_image_path)
                    else:
                        ref_image_path = video_path.replace(".mp4", ".jpg")
                    # if "flux" in ref_image_path: # temperaturelly supports the inputs from the flux
                    try:
                        # replacement_img_path =  ref_image_path
                        # replacement_img = load_image(ref_image_path)  # 假设有一个函数可以加载图片
                        # 使用cv2.IMREAD_UNCHANGED标志读取图片，以保留alpha通道
                        img = cv2.imread(ref_image_path, cv2.IMREAD_UNCHANGED)
                        assert img is not None, f"img is None, {ref_image_path}"
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = torch.from_numpy(img.astype(np.float32) / 255)  # (h, w, 3)
                            # print("img.shape", img.shape)
                            # print("cond_imgs.shape", cond_imgs.shape)
                        # test_img_paths[0] = ref_image_path
                        # results.update(test_imgs=test_imgs, test_img_paths=test_img_paths)

                        # ======== loading the reference smplx for images ==========
                        load_ref_smplx = False
                        if load_ref_smplx:
                            # if "flux" in ref_image_path:
                            smplx_smplify_path = from_video_to_get_ref_smplx(video_path)
                            # load json and get values
                            with open(smplx_smplify_path) as f:
                                data = json.load(f)

                            RT = torch.concatenate([ torch.Tensor(data['camera']['R']), torch.Tensor(data['camera']['t']).reshape(3,1)], dim=1)
                            RT = torch.cat([RT, torch.Tensor([[0,0,0,1]])], dim=0)


                            intri = torch.Tensor(data['camera']['focal'] + data['camera']['princpt'])

                            intri[[3,2]] = intri[[2,3]]
                            intri = intri * self.default_fxy_cxy[0,-1] / intri[-1]
                            
                            # 假设 smpl_param_data 是已经加载好的数据 
                            # (['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans', 'betas_save', 'nlf_smplx_betas', 'camera', 'img_path'])
                            smpl_param_data = data 

                            # 从字典中提取所需的数据
                            global_orient = np.array(smpl_param_data['root_pose']).reshape(1, -1)
                            body_pose = np.array(smpl_param_data['body_pose']).reshape(1, -1)
                            shape = np.array(smpl_param_data['betas_save']).reshape(1, -1)[:, :10]
                            left_hand_pose = np.array(smpl_param_data['lhand_pose']).reshape(1, -1)
                            right_hand_pose = np.array(smpl_param_data['rhand_pose']).reshape(1, -1)

                            # smpl_param_ref = np.concatenate([np.array([[1.]]), np.array([[0, 0, 0]]),
                            smpl_param_ref = np.concatenate([np.array([[1.]]), np.array(smpl_param_data['trans']).reshape(1,3),
                                global_orient,body_pose,
                                shape, left_hand_pose, right_hand_pose,
                                np.array(smpl_param_data['jaw_pose']).reshape(1, -1), np.array(smpl_param_data['leye_pose']).reshape(1, -1), np.array(smpl_param_data['reye_pose']).reshape(1, -1),
                                np.array(smpl_param_data['expr']).reshape(1, -1)[:,:10]], axis=1)


                            cond_poses[0] = RT         # RT
                            cond_intrinsics[0]  = intri   # fxfycxcy
                            cond_smpl_param_ref =  torch.Tensor(smpl_param_ref).reshape(-1)      # 189, combines 
                            if_use_smpl_param_ref = torch.Tensor([1])  # use the smpl_param_ref

                            # overwrite some datas
                            cond_imgs[0] = img
                            cond_img_paths[0] = ref_image_path

                    except (FileNotFoundError, json.JSONDecodeError, KeyError, Exception) as e:
                        # 记录错误信息到日志文件
                        # with open(self.log_file_path, 'a') as log_file:
                        #     log_file.write(f"{datetime.datetime.now()} {video_path} \n  - An error occurred: {str(e)}\n")
                        print(f"An error occurred: {e}")
                        
            ''' randomly crop the first image for augmentation'''
            if self.crop:
                # print("crop", cond_imgs[0].shape)
                if random.random() < 0.5:
                    # cond_imgs[0] = F.crop(cond_imgs[0], 0, 0, 512, 512)
                    crop_imgs = cond_imgs[0]
                    # 图像尺寸
                    h, w, _ = crop_imgs.shape

                    # 随机偏移量
                    random_offset_head = np.random.randint(-h//7, -h//8)
                    random_offset_body = np.random.randint(-h // 8, h // 8)

                    # head_joint, upper_body_joint
                    head_joint = [ w//2, h//7,]
                    upper_body_joint = [w//2, h//2, ]

                    # 计算裁剪区域
                    head_y = int(head_joint[1]) + random_offset_head
                    body_y = int(upper_body_joint[1]) + random_offset_body

                    # 确保裁剪区域在图像范围内
                    head_y = max(0, min(h, head_y))
                    body_y = max(0, min(h, body_y))

                    # 裁剪区域的高度和宽度
                    crop_height = body_y - head_y
                    crop_width =int(crop_height * 640 / 896)

                    # 确保裁剪区域在图像范围内
                    start_x = max(0, min(w - crop_width, int(w // 2 - crop_width // 2)))
                    end_x = start_x + crop_width
                    start_y = max(0, head_y)
                    end_y = min(h, body_y)

                    # 裁剪图像
                    cropped_img = crop_imgs[start_y:end_y, start_x:end_x]

            
                    padded_img = F.resize(cropped_img.permute(2, 0, 1), [h, w]).permute(1, 2, 0)

                    # save this img for debug
                    # Image.fromarray((padded_img.numpy() * 255).astype(np.uint8)).save(f"debug_crop.png")
                    # rescale the image for augmentation
                    cond_imgs[0] = random_scale_and_crop(padded_img, (0.8,1.2))
                else:
                    cond_imgs[0] = random_scale_and_crop(cond_imgs[0], (0.8,1.1))

            # =========== 数据格式转换 ===========
            # 1. 将图片从 (T,h,w,3) 转换为 (T,3,h,w)
            if cond_imgs is not None:
                cond_imgs = cond_imgs.permute(0, 3, 1, 2)  # (T, h, w, 3) -> (T, 3, h, w)
            
            # 2. 将smplx参数拆分成各个部分
            T = len(cond_inds)
            scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, \
                jaw_pose, leye_pose, reye_pose, expression = torch.split(
                    cond_smpl_param, 
                    [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=0)
            
            # 3. 将内参转换为内参矩阵
            cond_intrinsics_matrix = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
            cond_intrinsics_matrix[:, 0, 0] = cond_intrinsics[:, 0]  # fx
            cond_intrinsics_matrix[:, 1, 1] = cond_intrinsics[:, 1]  # fy
            cond_intrinsics_matrix[:, 0, 2] = cond_intrinsics[:, 2]  # cx
            cond_intrinsics_matrix[:, 1, 2] = cond_intrinsics[:, 3]  # cy
            
            # 4. 将smplx参数从1个复制到T份
            cond_smpl_params = []
            for i in range(T):
                cond_smpl_params.append({
                    # 'scale': scale,
                    'transl': transl,
                    'global_orient': global_orient,
                    'body_pose': pose.reshape(21, 3),
                    'betas': betas,
                    'left_hand_pose': left_hand_pose.reshape(15, 3),
                    'right_hand_pose': right_hand_pose.reshape(15, 3),
                    'jaw_pose': jaw_pose,
                    'leye_pose': leye_pose,
                    'reye_pose': reye_pose,
                    'expression': expression
                })
            
            results.update(
                cond_poses=cond_poses,
                cond_intrinsics=cond_intrinsics.to(torch.float32),
                cond_intrinsics_matrix=cond_intrinsics_matrix,  # 新增内参矩阵
                cond_img_paths=cond_img_paths, 
                cond_smpl_param=cond_smpl_param,
                cond_smpl_params=cond_smpl_params,  # 新增拆分后的smplx参数
                cond_smpl_param_ref=cond_smpl_param_ref,
                if_use_smpl_param_ref=if_use_smpl_param_ref)
            if cond_imgs is not None:
                results.update(cond_imgs=cond_imgs)
            if cond_norm is not None:
                results.update(cond_norm=cond_norm)

        if self.load_test_data and len(test_inds) > 0:
            # print("input_is_video", input_is_video)
            test_imgs, test_poses, test_intrinsics, test_img_paths, test_smpl_param, test_norm = \
                    gather_imgs(test_inds, poses, image_paths_or_video, smpl_params, 
                               load_imgs=self.load_imgs, load_norm=self.load_norm, 
                               center=self.center, radius=self.radius,
                               input_is_video=input_is_video)
            
            if test_intrinsics.shape[-1] == 3: # the old data format, which contains the value of camera center instead of fxfycxcy
                test_intrinsics = self.default_fxy_cxy.clone().repeat(test_intrinsics.shape[0], 1)

            # =========== 测试数据格式转换 ===========
            # 1. 将图片从 (T,h,w,3) 转换为 (T,3,h,w)
            if test_imgs is not None:
                test_imgs = test_imgs.permute(0, 3, 1, 2)  # (T, h, w, 3) -> (T, 3, h, w)
            
            # 2. 将smplx参数拆分成各个部分
            T_test = len(test_inds)
            test_scale, test_transl, test_global_orient, test_pose, test_betas, test_left_hand_pose, test_right_hand_pose, \
                test_jaw_pose, test_leye_pose, test_reye_pose, test_expression = torch.split(
                    test_smpl_param, 
                    [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=0)
            
            # 3. 将内参转换为内参矩阵
            test_intrinsics_matrix = torch.eye(3).unsqueeze(0).repeat(T_test, 1, 1)
            test_intrinsics_matrix[:, 0, 0] = test_intrinsics[:, 0]  # fx
            test_intrinsics_matrix[:, 1, 1] = test_intrinsics[:, 1]  # fy
            test_intrinsics_matrix[:, 0, 2] = test_intrinsics[:, 2]  # cx
            test_intrinsics_matrix[:, 1, 2] = test_intrinsics[:, 3]  # cy
            
            # 4. 将smplx参数从1个复制到T份
            test_smpl_params = []
            for i in range(T_test):
                test_smpl_params.append({
                    'scale': test_scale,
                    'transl': test_transl,
                    'global_orient': test_global_orient,
                    'body_pose': test_pose,
                    'betas': test_betas,
                    'left_hand_pose': test_left_hand_pose,
                    'right_hand_pose': test_right_hand_pose,
                    'jaw_pose': test_jaw_pose,
                    'leye_pose': test_leye_pose,
                    'reye_pose': test_reye_pose,
                    'expression': test_expression
                })

            results.update(
                test_poses=test_poses,
                test_intrinsics=test_intrinsics,
                test_intrinsics_matrix=test_intrinsics_matrix,  # 新增内参矩阵
                test_img_paths=test_img_paths,
                test_smpl_param=test_smpl_param,
                test_smpl_params=test_smpl_params)  # 新增拆分后的smplx参数
            if test_imgs is not None:
                results.update(test_imgs=test_imgs)
            if test_norm is not None:
                results.update(test_norm=test_norm)

    
        if self.test_pose_override is not None:
            results.update(test_poses=self.test_poses, test_intrinsics=self.test_intrinsics)
        return results

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, scene_id):
        try:
            # print('gettingingingingign')
            scene = self.parse_scene(scene_id)

        except:
            print("ERROR in parsing ", scene_id)
            scene = self.parse_scene(0)
        return scene
    
    def parse_scene_test(self, scene_id):
        scene = self.scenes[scene_id]
        input_is_video = False # flag of if the input is video, some operations should be different

        
        # =========== loading the params ===========
        param = np.load(scene['param_path'], allow_pickle=True).item()
        scene.update(param)
        print(scene.keys())
        # import ipdb; ipdb.set_trace()
        if scene['image_paths'] is None:
            input_is_video = True
            video_path = scene['video_path']
            try:
                image_paths_or_video = read_frames(scene['video_path'])
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error in reading the video : {scene['video_path'].replace('result.mp4', 'output.mp4')}")
                image_paths_or_video = read_frames(scene['video_path'])

        if not input_is_video:
            image_paths_or_video = scene['image_paths']
        scene_name = f"{scene_id:0>4d}" #  image_paths[0].split('/')[-3]
        results = dict(
            scene_id=[scene_id],
            scene_name=
                '{:04d}'.format(scene_id) if self.scene_id_as_name else scene_name,
                # cpu_only=True
                )
        
        # if not self.code_only:
        poses = scene['poses']
        smpl_params = scene['smpl_params']

        # if input_is_video:
        #     num_imgs = len(video)
        # else:
        num_imgs = len(image_paths_or_video)
        # front_view = num_imgs // 4
        # randonly / specificically select the views of output
        smplx_cam_rotate = smpl_params[4: 7] #get global orient # 1, 3, 63, 10
        # smpl_params[70:80] = torch.rand_like(smpl_params[70:80]); print("error !! need to delete this rand betas in avatarnet:287") # get betas
        front_view = find_front_camera_by_rotation(poses, smplx_cam_rotate) # inputs camera poses and smplx poses
        if self.allow_k_angles_near_the_front > 0:
            allow_n_views_near_the_front =  round(self.allow_k_angles_near_the_front / 360 * num_imgs)
            new_front_view = random.randint(-allow_n_views_near_the_front, allow_n_views_near_the_front) + front_view
            if new_front_view >= num_imgs:
                new_front_view = new_front_view - num_imgs
            elif new_front_view < 0:
                new_front_view = new_front_view + num_imgs
            front_view = new_front_view
            print("changes the front views ranges", front_view, "+-", allow_n_views_near_the_front)

        num_train_imgs = num_imgs
        test_inds = torch.Tensor( list(range(num_imgs)))
        cond_inds = np.concatenate([np.array([front_view]),test_inds]).astype(np.int64) # first for input, second for supervised
        test_inds = cond_inds.tolist()
        # if self.specific_observation_num: # yy note: if specific_observation_num is not None, then remove the test_inds
        #     test_inds = []
        # else:
        # for cond_ind in cond_inds:
        #     test_inds.remove(cond_ind)
        # cond_inds = cond_inds

        cond_smpl_param_ref = torch.zeros([189])
        if_use_smpl_param_ref = torch.Tensor([1]) # 默认使用ref smpl, 
        if self.load_cond_data and len(cond_inds) > 0:
            # cond_imgs, cond_poses, cond_intrinsics, cond_img_paths, cond_smpl_param, cond_norm = gather_imgs(cond_inds)
            cond_imgs, cond_poses, cond_intrinsics, cond_img_paths, cond_smpl_param, cond_norm = \
                gather_imgs(cond_inds, poses, image_paths_or_video, smpl_params, 
                           load_imgs=self.load_imgs, load_norm=self.load_norm, 
                           center=self.center, radius=self.radius,
                           input_is_video=input_is_video)
            cond_smpl_param_ref = cond_smpl_param.clone() # the smpl_param_ref for the reference images
            if cond_intrinsics.shape[-1] == 3: # the old data format, which contains the value of camera center instead of fxfycxcy
                cond_intrinsics = self.default_fxy_cxy.clone().repeat(cond_intrinsics.shape[0], 1)

            if input_is_video:
                cond_img_paths = [f"{video_path[:-4]}_{i:0>4d}.png" for i in  range(self.specific_observation_num)] # Replace the .mp4 into index.png
            
            # =========== 数据格式转换 ===========
            # 1. 将图片从 (T,h,w,3) 转换为 (T,3,h,w)
            if cond_imgs is not None:
                cond_imgs = cond_imgs.permute(0, 3, 1, 2)  # (T, h, w, 3) -> (T, 3, h, w)
            
            # 2. 将smplx参数拆分成各个部分
            T = len(cond_inds)
            scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, \
                jaw_pose, leye_pose, reye_pose, expression = torch.split(
                    cond_smpl_param, 
                    [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=0)
            
            # 3. 将内参转换为内参矩阵
            cond_intrinsics_matrix = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
            cond_intrinsics_matrix[:, 0, 0] = cond_intrinsics[:, 0]  # fx
            cond_intrinsics_matrix[:, 1, 1] = cond_intrinsics[:, 1]  # fy
            cond_intrinsics_matrix[:, 0, 2] = cond_intrinsics[:, 2]  # cx
            cond_intrinsics_matrix[:, 1, 2] = cond_intrinsics[:, 3]  # cy
            
            # 4. 将smplx参数从1个复制到T份
            cond_smpl_params = []
            for i in range(T):
                cond_smpl_params.append({
                    'scale': scale,
                    'transl': transl,
                    'global_orient': global_orient,
                    'body_pose': pose,
                    'betas': betas,
                    'left_hand_pose': left_hand_pose,
                    'right_hand_pose': right_hand_pose,
                    'jaw_pose': jaw_pose,
                    'leye_pose': leye_pose,
                    'reye_pose': reye_pose,
                    'expression': expression
                })
                    
            results.update(
                cond_poses=cond_poses,
                cond_intrinsics=cond_intrinsics.to(torch.float32),
                cond_intrinsics_matrix=cond_intrinsics_matrix,  # 新增内参矩阵
                cond_img_paths=cond_img_paths, 
                cond_smpl_param=cond_smpl_param,
                cond_smpl_params=cond_smpl_params,  # 新增拆分后的smplx参数
                cond_smpl_param_ref=cond_smpl_param_ref,
                if_use_smpl_param_ref=if_use_smpl_param_ref)
            if cond_imgs is not None:
                results.update(cond_imgs=cond_imgs)
            if cond_norm is not None:
                results.update(cond_norm=cond_norm)

        if self.load_test_data and len(test_inds) > 0:
            print("input_is_video", input_is_video)
            test_imgs, test_poses, test_intrinsics, test_img_paths, test_smpl_param, test_norm = \
                    gather_imgs(test_inds, poses, image_paths_or_video, smpl_params, 
                               load_imgs=self.load_imgs, load_norm=self.load_norm, 
                               center=self.center, radius=self.radius,
                               input_is_video=input_is_video)
            
            if test_intrinsics.shape[-1] == 3: # the old data format, which contains the value of camera center instead of fxfycxcy
                test_intrinsics = self.default_fxy_cxy.clone().repeat(test_intrinsics.shape[0], 1)

            # =========== 测试数据格式转换 ===========
            # 1. 将图片从 (T,h,w,3) 转换为 (T,3,h,w)
            if test_imgs is not None:
                test_imgs = test_imgs.permute(0, 3, 1, 2)  # (T, h, w, 3) -> (T, 3, h, w)
            
            # 2. 将smplx参数拆分成各个部分
            T_test = len(test_inds)
            test_scale, test_transl, test_global_orient, test_pose, test_betas, test_left_hand_pose, test_right_hand_pose, \
                test_jaw_pose, test_leye_pose, test_reye_pose, test_expression = torch.split(
                    test_smpl_param, 
                    [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=0)
            
            # 3. 将内参转换为内参矩阵
            test_intrinsics_matrix = torch.eye(3).unsqueeze(0).repeat(T_test, 1, 1)
            test_intrinsics_matrix[:, 0, 0] = test_intrinsics[:, 0]  # fx
            test_intrinsics_matrix[:, 1, 1] = test_intrinsics[:, 1]  # fy
            test_intrinsics_matrix[:, 0, 2] = test_intrinsics[:, 2]  # cx
            test_intrinsics_matrix[:, 1, 2] = test_intrinsics[:, 3]  # cy
            
            # 4. 将smplx参数从1个复制到T份
            test_smpl_params = []
            for i in range(T_test):
                test_smpl_params.append({
                    'scale': test_scale,
                    'transl': test_transl,
                    'global_orient': test_global_orient,
                    'body_pose': test_pose,
                    'betas': test_betas,
                    'left_hand_pose': test_left_hand_pose,
                    'right_hand_pose': test_right_hand_pose,
                    'jaw_pose': test_jaw_pose,
                    'leye_pose': test_leye_pose,
                    'reye_pose': test_reye_pose,
                    'expression': test_expression
                })

            results.update(
                test_poses=test_poses,
                test_intrinsics=test_intrinsics,
                test_intrinsics_matrix=test_intrinsics_matrix,  # 新增内参矩阵
                test_img_paths=test_img_paths,
                test_smpl_param=test_smpl_param,
                test_smpl_params=test_smpl_params)  # 新增拆分后的smplx参数
            if test_imgs is not None:
                results.update(test_imgs=test_imgs)
            if test_norm is not None:
                results.update(test_norm=test_norm)

    
        if self.test_pose_override is not None:
            results.update(test_poses=self.test_poses, test_intrinsics=self.test_intrinsics)
        return results


def gather_imgs(img_ids, poses, image_paths_or_video, smpl_params, load_imgs=True, load_norm=False, center=None, radius=None, input_is_video=False):
    imgs_list = [] if load_imgs else None
    norm_list = [] if load_norm else None
    poses_list = []
    cam_centers_list = []
    img_paths_list = []
   
    for img_id in img_ids:
        pose = poses[img_id][0]
        cam_centers_list.append((poses[img_id][1]).to(torch.float)) # (C)
        c2w = pose.to(torch.float)#torch.FloatTensor(pose) # 虽然是c2w但其实存的值应该是w2c (R|T)
        cam_to_ndc = torch.cat(
            [c2w[:3, :3], (c2w[:3, 3:] - center[:, None]) / radius[:, None]], dim=-1)
        poses_list.append(
            torch.cat([
                cam_to_ndc,
                cam_to_ndc.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            ], dim=-2))
        if input_is_video:
            # img_paths_list.append(video[img_id])
            img = image_paths_or_video[img_id]
            # for img in imgs: # add the ajustment to make the color > [250,250,250] to be white
            mask_white = np.all(img[:,:,:3] > 250, axis=-1, keepdims=False)
            # Image.fromarray(img).save(f"debug.png")
            # Image.fromarray(mask_white).save(f"debug_mask.png")
            img[mask_white] = [255, 255, 255]
            # Image.fromarray(img).save(f"debug_afmask.png")
            img = torch.from_numpy(img.astype(np.float32) / 255)  # (h, w, 3)
           
            imgs_list.append(img)
        else:
            img_paths_list.append(image_paths_or_video[img_id])
            if load_imgs:
                # img = mmcv.imread(image_paths[img_id], channel_order='rgb')
                                        
                # 使用cv2.IMREAD_UNCHANGED标志读取图片，以保留alpha通道
                print("Loading, .......", image_paths_or_video[img_id])
                print("Loading, .......", image_paths_or_video[img_id])
                print("Loading, .......", image_paths_or_video[img_id])
                img = cv2.imread(image_paths_or_video[img_id], cv2.IMREAD_UNCHANGED)
                print("img.shape", img.shape)
                # 将透明像素的RGB值设置为白色（255, 255, 255）
                img[img[..., 3] == 0] = [255, 255, 255, 255]
                img = img[..., :3]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img.astype(np.float32) / 255)  # (h, w, 3)
                imgs_list.append(img)
        if load_norm: # have not support the input type is video
            norm = cv2.imread(image_paths_or_video[img_id].replace('rgb', 'norm'), cv2.IMREAD_UNCHANGED)
            norm = cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)
            norm = torch.from_numpy(norm.astype(np.float32) / 255)
            norm_list.append(norm)
    poses_list = torch.stack(poses_list, dim=0)  # (n, 4, 4)
    cam_centers_list = torch.stack(cam_centers_list, dim=0)
    if load_imgs:
        imgs_list = torch.stack(imgs_list, dim=0)  # (n, h, w, 3)
    if load_norm:
        norm_list = torch.stack(norm_list, dim=0)
    return imgs_list, poses_list, cam_centers_list, img_paths_list, smpl_params, norm_list

from scipy.spatial.transform import Rotation as R
def calculate_angle(vector1, vector2):
    unit_vector1 = vector1 / torch.linalg.norm(vector1)
    unit_vector2 = vector2 / torch.linalg.norm(vector2)
    dot_product = torch.dot(unit_vector1, unit_vector2)
    angle = torch.arccos(dot_product)
    return angle
def axis_angle_to_rotation_matrix(axis_angle):
    if_input_is_torch = torch.is_tensor(axis_angle)
    if if_input_is_torch:
        dtype_torch = axis_angle.dtype
        axis_angle = axis_angle.numpy()
        
    r = R.from_rotvec(axis_angle)
    rotation_matrix = r.as_matrix()
    if if_input_is_torch:
        rotation_matrix = torch.from_numpy(rotation_matrix).to(dtype_torch)

    return rotation_matrix
# def find_front_camera_by_global_orient(global_orient, camera_direction):
#     front_direction = np.array([0, 0, -1])  # 人体正面方向
#     min_angle = float('inf')
#     front_camera_idx = -1

#     for idx, global_orient in enumerate(global_orient_list):
        

#         angle = calculate_angle(body_direction, front_direction)
#         if angle < min_angle:
#             min_angle = angle
#             front_camera_idx = idx

#     return front_camera_idx
def find_front_camera_by_rotation(poses, global_orient_human):
    # front_direction = global_orient_human  # 人体正面方向
    rotation_matrix = axis_angle_to_rotation_matrix(global_orient_human)
    front_direction = rotation_matrix @ torch.Tensor([0, 0, -1])  # 人体的朝向
    min_angle = float('inf')
    front_camera_idx = -1

    for idx, pose in enumerate(poses):
        rotation_matrix = pose[0][:3, :3]
        camera_direction = rotation_matrix @ torch.Tensor([0, 0, 1])  # 相机的朝向
        angle = calculate_angle(camera_direction, front_direction).to(camera_direction.dtype)
        if angle < min_angle:
            min_angle = angle
            front_camera_idx = idx

    return front_camera_idx

def read_frames(video_path):
    # print(11111, flush=True)
    container = av.open(video_path)
    # print(22222, flush=True)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            # image = Image.frombytes(
            #     "RGB",
            #     (frame.width, frame.height),
            #     frame.to_rgb().to_ndarray(),
            # )
            image =  frame.to_rgb().to_ndarray()
            frames.append(image)

    return frames


def prepare_camera( resolution_x, resolution_y, num_views=24, stides=1):
    # resolution_x = 640
    # resolution_y = 896
    import math
    focal_length = 40 #80
    sensor_width = 32

    # # 创建 Pyrender 相机
    # camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=aspect_ratio)
    focal_length = focal_length * (resolution_y/sensor_width)

    K = np.array(
        [[focal_length, 0, resolution_x//2],
        [0, focal_length, resolution_y//2],
        [0, 0, 1]]
    )
    # print("update!! the camera intrisic is error 0819")
    def look_at(camera_position, target_position, up_vector):  # colmap +z forward, +y down
        forward = -(camera_position - target_position) / np.linalg.norm(camera_position - target_position)
        right = np.cross(up_vector, forward)
        up = np.cross(forward, right)
        return np.column_stack((right, up, forward))
    camera_pose_list = []
    for frame_idx in range(0, num_views, stides):
        # 设置相机的位置和方向
        camera_dist = 1.5 #3 #1.2 * 2
        phi = math.radians(90)
        theta = (frame_idx / num_views) * math.pi * 2
        camera_location = np.array(
            [camera_dist * math.sin(phi) * math.cos(theta),
            
            camera_dist * math.cos(phi),
            -camera_dist * math.sin(phi) * math.sin(theta),]
            )
        # print(camera_location)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_location
        # print("camera_location", camera_location)

        # from smplx import look_at


        # 设置相机位置和目标位置
        camera_position = camera_location
        target_position = np.array([0.0, 0.0, 0.0])

        # 计算相机的旋转矩阵，使其朝向目标
        # up_vector = np.array([0.0, 1.0, 0.0])
        up_vector = np.array([0.0, -1.0, 0.0]) # colmap
        rotation_matrix = look_at(camera_position, target_position, up_vector)

        # 更新相机的位置和旋转
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = camera_position
        camera_pose_list.append(camera_pose)
    return K, camera_pose_list


def from_video_to_get_ref_smplx(video_path):
    # 分解路径
    video_dir = os.path.dirname(video_path)
    video_name = video_dir.split("/")[-1] # 视频文件夹名称
    
    # 替换视频目录为 smplify 目录
    if "flux" in video_dir:
        smplify_dir = video_dir.replace('/videos/', '/smplx_smplify/')
    elif "DeepFashion" in video_dir:
        smplify_dir = video_dir.replace('/video/', '/smplx_smplify/').replace("A_pose_", "")
    
    # # 获取视频文件名（不包括扩展名）
    # video_name = os.path.splitext(video_file)[0]
    
    # 构建 JSON 文件路径
    if smplify_dir[-1] == '/': smplify_dir = smplify_dir[:-1]
    json_path = smplify_dir+".json" #os.path.join(smplify_dir, f"{video_name}.json")
    
    return json_path

def random_scale_and_crop(image: torch.Tensor, scale_range=(0.8, 1.2)) -> torch.Tensor:
    """
    Randomly scale the input image and crop/pad to maintain original size.

    Args:
        image: Input image tensor of shape [H, W, 3]
        scale_range: Range for scaling factor, default (0.8, 1.2)

    Returns:
        Scaled and cropped/padded image tensor of shape [H, W, 3]
    """
    is_numpy = False
    if not torch.is_tensor(image):
        image = torch.from_numpy(image)
        is_numpy = True
    # 获取图像的高度和宽度
    h, w = image.shape[:2]

    # 生成随机缩放因子
    scale_factor = random.uniform(*scale_range)

    # 计算新的高度和宽度
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # 使用 torchvision.transforms.functional.resize 进行缩放
    scaled_image = F.resize(image.permute(2, 0, 1), [new_h, new_w]).permute(1, 2, 0)

    # 如果缩放后的图像比原图大，进行居中裁剪
    if new_h > h or new_w > w:
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        scaled_image = scaled_image[top:top + h, left:left + w]
    else:
        # 如果缩放后的图像比原图小，进行居中填充
        padded_image = torch.ones((h, w, 3), dtype=image.dtype)
        top = h-new_h #(h - new_h) // 2 # H不应该居中
        left = (w - new_w) // 2
        padded_image[top:top + new_h, left:left + new_w] = scaled_image
        scaled_image = padded_image
    if is_numpy:
        scaled_image = scaled_image.numpy()
    return scaled_image

        

if __name__ == "__main__":


    import os

          

    params = {
        "data_prefix": None,
        "cache_path":  ListConfig([ 
            "/disk2/liubingqi/data/IDOL/processed_data/flux_batch8_train_17483.npy",
            "/disk2/liubingqi/data/IDOL/processed_data/flux_batch9_train_13425.npy"
            # "/home/liubingqi/work/data/IDOL/processed_data/flux_batch1_5000_train_145_local.npy",
            # "/home/liubingqi/work/data/IDOL/processed_data/flux_batch2_train_145_local.npy"
        ]),
        "specific_observation_num": 8,
        "better_range": True,
        "first_is_front": True,
        "if_include_video_ref_img": False,
        "prob_include_video_ref_img": 0.0,
        "img_res": [640, 896],
        'test_mode': False,
        'load_test_data': True,
        'allow_k_angles_near_the_front': 90,
    }

    data = AvatarDataset(**params)

    sample = data[20000]
    print(sample.keys())
    for k, v in sample['cond_smpl_params'][0].items():
        print(k, v.shape)
    # print(sample['cond_imgs'].shape)
    # print(sample['cond_smpl_param'].shape)

    # print(sample['cond_smpl_params'])
    # print(sample['cond_intrinsics_matrix'].shape)
    # print(sample['cond_poses'].shape)

    from src.utils.graphics_utils import init_smplx_model, SimpleMeshRenderer
    smplx_model = init_smplx_model(device=sample['cond_imgs'].device,
                                   num_expression_coeffs=10,
                                   num_betas=10,
                                   num_pca_comps=12,
                                   flat_hand_mean=False,
                                   use_pca=False,
                                   create_body_pose=False,
                                   create_betas=False,
                                   create_global_orient=False)
    renderer = SimpleMeshRenderer(smplx_model.faces, 896, 640)

    K = sample['cond_intrinsics_matrix'][0].reshape(3, 3)
    E = sample['cond_poses'][0].reshape(4, 4)
    device = next(smplx_model.parameters()).device

    params = {}
    for key, tensor in sample['cond_smpl_params'][0].items():
        params[key] = tensor.to(device).reshape(1, -1)

    with torch.no_grad():
        out = smplx_model(global_orient=params['global_orient'], body_pose=params['body_pose'], betas=params['betas'], left_hand_pose=params['left_hand_pose'],
               right_hand_pose=params['right_hand_pose'], jaw_pose=params['jaw_pose'], leye_pose=params['leye_pose'], reye_pose=params['reye_pose'],
               expression=params['expression'])
        verts = out.vertices.detach() 
    
    rendered = renderer.render_mesh(verts[0], sample['cond_imgs'][0].cpu().numpy().transpose(1, 2, 0), K, E)
    rendered_uint8 = (np.clip(rendered, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(rendered_uint8).save("rendered.png")
    ori_img = sample['cond_imgs'][0].cpu().numpy().transpose(1, 2, 0)
    ori_img = (np.clip(ori_img, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(ori_img).save("ori_img.png")
        
