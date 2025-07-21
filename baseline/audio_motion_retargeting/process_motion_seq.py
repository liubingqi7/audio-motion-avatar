import numpy as np
import json
import os
import glob
from scipy.spatial.transform import Rotation as R

input_dir = "/home/liubingqi/work/audio_motion_avatar/baseline/PantoMatrix/examples/motion"
output_base_dir = "tmp_data/output_motion_pretrain"

npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
print(f"找到 {len(npz_files)} 个npz文件需要处理")

for npz_file in npz_files:
    file_basename = os.path.basename(npz_file)
    file_name_without_ext = os.path.splitext(file_basename)[0]
    output_name = file_name_without_ext.replace("res_", "")
    
    print(f"正在处理文件: {file_basename}")
    
    # 加载 npz 文件
    data = np.load(npz_file)
    
    # 获取数据
    betas = data["betas"][:10].tolist()  # 取前10维的betas
    poses = data["poses"]  # (num_frames, 165)
    trans = data["trans"]  # (num_frames, 3)
    
    num_frames = poses.shape[0]
    
    # 创建输出目录
    output_dir = os.path.join(output_base_dir, f"{output_name}_res_short/smplx_params")
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每一帧
    for i in range(num_frames):
        # 正确处理root_pose，绕x轴旋转180度
        # 获取原始的轴角表示
        root_pose_orig = poses[i, 0:3].copy()
        
        # 将原始轴角转换为旋转矩阵
        rot_orig = R.from_rotvec(root_pose_orig)
        
        # 创建绕x轴旋转180度的旋转矩阵
        rot_x_180 = R.from_rotvec([np.pi, 0, 0])
        
        # 组合两个旋转（先应用原始旋转，再绕x轴旋转180度）
        rot_combined = rot_x_180 * rot_orig
        
        # 将组合后的旋转转换回轴角表示
        root_pose_final = rot_combined.as_rotvec()
        
        frame_data = {
            "betas": betas,
            "root_pose": root_pose_final.tolist(),
            "body_pose": poses[i, 3:66].reshape(21, 3).tolist(),
            "jaw_pose": poses[i, 66:69].tolist(),
            "leye_pose": poses[i, 69:72].tolist(),
            "reye_pose": poses[i, 72:75].tolist(),
            "lhand_pose": poses[i, 75:120].reshape(15, 3).tolist(),
            "rhand_pose": poses[i, 120:165].reshape(15, 3).tolist(),
            # "trans": (trans[i]).tolist(),
            "trans": [0.02096693404018879, 0.3983211815357208, 2.393183946609497],
            "focal": [1000, 1000],
            "princpt": [345.0, 614.0],
            "img_size_wh": [691, 1229],
            "pad_ratio": 0
        }
        
        output_path = os.path.join(output_dir, f"{i:06d}.json")
        with open(output_path, "w") as f:
            json.dump(frame_data, f, indent=2)
        
        if i % 50 == 0:
            print(f"已保存第 {i} 帧到 {output_path}")

        # if i > 400:
        #     break
    
    print(f"文件 {file_basename} 的所有 {num_frames} 帧已处理完成并保存到 '{output_dir}' 目录。")