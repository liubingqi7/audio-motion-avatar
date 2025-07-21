import os
import torch
import numpy as np
import glob
import json
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from PIL import Image

class GaussianAudioDataset(Dataset):
    def __init__(self, root_dir="/home/liubingqi/work/data/clip_2m52s_to_3m10s", 
                 audio_file=None, wav2vec2_model_path="/home/liubingqi/work/audio2avatar/gaussian_avatar/wav2vec2-base-960h",
                 sample_rate=16000, clip_length=8):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "imgs_png")
        self.smplx_params_dir = os.path.join(root_dir, "smplx_params")
        self.clip_length = clip_length
        
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))

        if audio_file is None:
            audio_files = glob.glob(os.path.join(root_dir, "*.mp3")) + glob.glob(os.path.join(root_dir, "*.wav"))
            if audio_files:
                self.audio_file = audio_files[0]
            else:
                raise FileNotFoundError(f"未在 {root_dir} 找到音频文件")
        else:
            self.audio_file = audio_file

        self.wav2vec2_model_path = wav2vec2_model_path
            
        self.sample_rate = sample_rate
        self.audio_features = self._extract_audio_features()
        print(f"数据集初始化完成，共有 {len(self.image_files)} 帧数据")

    def _extract_audio_features(self):
        print(f"正在从 {self.audio_file} 提取音频特征...")
        
        waveform, sr = torchaudio.load(self.audio_file)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_model_path)
        model = Wav2Vec2Model.from_pretrained(self.wav2vec2_model_path)
        
        frames_count = len(self.image_files)
        audio_duration = waveform.shape[1] / self.sample_rate
        
        estimated_frame_rate = 30
        estimated_video_duration = frames_count / estimated_frame_rate
        
        if audio_duration > estimated_video_duration:
            print(f"音频时长 {audio_duration:.2f} 秒大于视频时长 {estimated_video_duration:.2f} 秒，进行裁剪")
            samples_to_keep = int(estimated_video_duration * self.sample_rate)
            waveform = waveform[:, :samples_to_keep]
            audio_duration = estimated_video_duration
            print(f"音频已裁剪至 {audio_duration:.2f} 秒，以匹配 {frames_count} 帧高斯模型")
          
        frame_duration = audio_duration / frames_count
        
        features = []
        
        for start_idx in range(0, frames_count, self.clip_length):
            end_idx = min(start_idx + self.clip_length, frames_count)
            
            start_time = start_idx * frame_duration
            end_time = end_idx * frame_duration
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            end_sample = min(end_sample, waveform.shape[1])
            
            if start_sample >= end_sample:
                start_sample = max(0, waveform.shape[1] - int((end_idx - start_idx) * frame_duration * self.sample_rate))
                end_sample = waveform.shape[1]
            
            clip_waveform = waveform[:, start_sample:end_sample]
            
            inputs = processor(clip_waveform.numpy().squeeze(), sampling_rate=self.sample_rate, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            hidden_states = outputs.last_hidden_state  # [1, T, D]
            
            time_steps = hidden_states.shape[1]
            frames_in_clip = end_idx - start_idx
            
            steps_per_frame = max(1, time_steps // frames_in_clip)
            
            for i in range(frames_in_clip):
                frame_start = min(i * steps_per_frame, time_steps - 1)
                frame_end = min((i + 1) * steps_per_frame, time_steps)
                
                if frame_start < frame_end:
                    frame_features = hidden_states[:, frame_start:frame_end, :].mean(dim=1).squeeze().numpy()
                else:
                    frame_features = hidden_states[:, frame_start:frame_start+1, :].squeeze().numpy()
                
                features.append(frame_features)

        if len(features) < frames_count:
            print(f"音频特征数量不足，需要扩展到 {frames_count} 个")
            last_feature = features[-1]
            features.extend([last_feature] * (frames_count - len(features)))
        elif len(features) > frames_count:
            print(f"音频特征数量超过 {frames_count} 个，需要裁剪到 {frames_count} 个")
            features = features[:frames_count]
            
        print(f"成功提取 {len(features)} 个音频特征")
        return features

    def _load_smplx_params(self, index):
        """加载SMPLX参数"""
        frame_id = os.path.basename(self.image_files[index]).split('.')[0]
        smplx_file = os.path.join(self.smplx_params_dir, f"{int(frame_id):05d}.json")
        
        with open(smplx_file, 'r') as f:
            smplx_params = json.load(f)

        smpl_params = {
            'betas': smplx_params['betas'],
            'transl': smplx_params['trans'],
            'global_orient': smplx_params['root_pose'],
            'body_pose': smplx_params['body_pose'],
            'left_hand_pose': smplx_params['lhand_pose'],
            'right_hand_pose': smplx_params['rhand_pose'],
            'jaw_pose': smplx_params['jaw_pose'],
            'leye_pose': smplx_params['leye_pose'], 
            'reye_pose': smplx_params['reye_pose'],
            'pad_ratio': smplx_params.get('pad_ratio', 0.0),
            'focal': smplx_params['focal'],
            'princpt': smplx_params['princpt'],
            'expression': [0.0] * 10,
        }
        return smpl_params
            
    def __len__(self):
        return max(1, len(self.image_files) - self.clip_length * 2 + 1)
        # return 50

    def __getitem__(self, index):
        clip_data = []
        
        for i in range(index, min(index + self.clip_length * 2, len(self.image_files)), 2):
            # print(f"正在处理第 {i} 帧...")
            image_file = self.image_files[i]

            smplx_params = self._load_smplx_params(i)
            smplx_params = {k: torch.FloatTensor(v) if isinstance(v, (list, np.ndarray)) else v 
                           for k, v in smplx_params.items()}
            
            extrinsic, intrinsic, bg_color = self._load_camera(smplx_params)
            extrinsic = torch.as_tensor(extrinsic, dtype=torch.float32)
            intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32)
            bg_color = torch.as_tensor(bg_color, dtype=torch.float32)
            
            render_h, render_w = int(intrinsic[1, 2] * 2), int(
                intrinsic[0, 2] * 2
            )
            
            audio_feature = torch.as_tensor(self.audio_features[i], dtype=torch.float32)

            frame_id = os.path.basename(image_file).split('.')[0]
            img_path = os.path.join(self.root_dir, "imgs_png", f"{int(frame_id):05d}.png")
            mask_path = os.path.join(self.root_dir, "samurai_seg", f"{int(frame_id):05d}.png")
            if not os.path.exists(img_path):
                print(f"警告: 找不到图片文件 {img_path}，使用空图像")
                image = torch.zeros((3, 512, 512), dtype=torch.float32)
            else:
                image = torch.from_numpy(np.array(Image.open(img_path).convert("RGB"))).permute(2, 0, 1).float() / 255.0
            
            if not os.path.exists(mask_path):
                print(f"警告: 找不到mask文件 {mask_path}，使用全1 mask")
                mask = torch.ones((1, image.shape[1], image.shape[2]), dtype=torch.float32)
            else:
                mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L"))).unsqueeze(0).float() / 255.0
            
            masked_image = image * mask + (1 - mask)

            pad_ratio = float(smplx_params.get('pad_ratio', 0.0))
            masked_image = pad_image(masked_image, pad_ratio)

            if mask.sum() > 0:
                mask_np = mask.squeeze().numpy()
                rows = np.any(mask_np, axis=1)
                cols = np.any(mask_np, axis=0)
                
                if np.any(rows) and np.any(cols):
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    
                    height = y_max - y_min
                    width = x_max - x_min
                    
                    padding_y = int(height * 0.2)
                    padding_x = int(width * 0.2)
                    
                    y_min = max(0, y_min - padding_y)
                    y_max = min(mask_np.shape[0] - 1, y_max + padding_y)
                    x_min = max(0, x_min - padding_x)
                    x_max = min(mask_np.shape[1] - 1, x_max + padding_x)
                    
                    crop_size = max(y_max - y_min, x_max - x_min)
                    center_y = (y_min + y_max) // 2
                    center_x = (x_min + x_max) // 2
                    
                    y_min = max(0, center_y - crop_size // 2)
                    y_max = min(mask_np.shape[0] - 1, center_y + crop_size // 2)
                    x_min = max(0, center_x - crop_size // 2)
                    x_max = min(mask_np.shape[1] - 1, center_x + crop_size // 2)
                    
                    cropped_image = masked_image[:, y_min:y_max+1, x_min:x_max+1]
                    cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]
                else:
                    cropped_image = masked_image
                    cropped_mask = mask
            else:
                cropped_image = masked_image
                cropped_mask = mask

            c, h, w = cropped_image.shape
            max_size = max(h, w)
            
            h_pad = max_size - h
            w_pad = max_size - w
            
            pad_size = (
                w_pad // 2,                    
                max_size - (w + w_pad // 2),   
                h_pad // 2,                    
                max_size - (h + h_pad // 2),   
            )
            
            square_image = torch.nn.functional.pad(
                cropped_image.unsqueeze(0), 
                pad_size, 
                value=1
            ).squeeze(0)
            
            if max_size != 1024:
                cropped_image = torch.nn.functional.interpolate(
                    square_image.unsqueeze(0), 
                    size=(1024, 1024), 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(0)
                
            processed_smplx_params = {}
            for k, v in smplx_params.items():
                if isinstance(v, (list, np.ndarray)):
                    processed_smplx_params[k] = torch.FloatTensor(v)
                elif isinstance(v, (int, float)):
                    processed_smplx_params[k] = torch.tensor(v, dtype=torch.float32)
                else:
                    processed_smplx_params[k] = v

            clip_data.append({
                'smplx_params': processed_smplx_params,
                'audio_feature': audio_feature,
                'image': masked_image,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'bg_color': bg_color,
                'cropped_image': cropped_image,
            })
        
        if len(clip_data) < self.clip_length:
            last_frame = clip_data[-1]
            for _ in range(self.clip_length - len(clip_data)):
                clip_data.append(last_frame)

        stacked_data = {
            'smplx_params': {k: torch.stack([frame['smplx_params'][k] for frame in clip_data]) 
                            for k in clip_data[0]['smplx_params'].keys() 
                            if isinstance(clip_data[0]['smplx_params'][k], torch.Tensor)},
            'audio_feature': torch.stack([frame['audio_feature'] for frame in clip_data]),
            'images': torch.stack([frame['image'] for frame in clip_data]),
            'camera': {
                'intrinsic': torch.stack([frame['intrinsic'] for frame in clip_data]),
                'extrinsic': torch.stack([frame['extrinsic'] for frame in clip_data])
            },
            'bg_color': torch.stack([frame['bg_color'] for frame in clip_data]),
            'cropped_images': torch.stack([frame['cropped_image'] for frame in clip_data]),
            'batch_id': index,
        }
        
        return stacked_data

    def _load_camera(self, smplx_params):
        smplx_params = {
            k: torch.FloatTensor(v) if isinstance(v, (list, np.ndarray)) else torch.tensor(v, dtype=torch.float32) if isinstance(v, (int, float)) else v
            for k, v in smplx_params.items()
            if "pad_ratio" not in k
        }

        c2w, intrinsic = _load_pose(smplx_params)
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        return c2w, intrinsic, bg_color

def _load_pose(pose):
    intrinsic = torch.eye(3)
    intrinsic[0, 0] = pose["focal"][0]
    intrinsic[1, 1] = pose["focal"][1]
    intrinsic[0, 2] = pose["princpt"][0]
    intrinsic[1, 2] = pose["princpt"][1]
    intrinsic = intrinsic.float()

    c2w = torch.eye(4)
    c2w = c2w.float()

    return c2w, intrinsic

def pad_image(image, pad_ratio):
    c, h, w = image.shape
    padded_h = int(h * (1 + pad_ratio))
    padded_w = int(w * (1 + pad_ratio))
    
    padded_image = torch.ones((c, padded_h, padded_w), dtype=image.dtype)
    
    offset_h = (padded_h - h) // 2
    offset_w = (padded_w - w) // 2
    
    padded_image[:, offset_h:offset_h + h, offset_w:offset_w + w] = image
    
    return padded_image

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = GaussianAudioDataset()
    print(len(dataset))
    data = dataset[100]
    print(data.keys())
    print(data['images'].shape)
    print(data['audio_feature'].shape)
    print(data['smplx_params'].keys())
    print(data['camera']['intrinsic'].shape)
    print(data['camera']['extrinsic'].shape)
    for i in range(len(data['cropped_images'])):
        print(data['cropped_images'][i].shape)
        
        cropped_img = data['cropped_images'][i].permute(1, 2, 0).numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(cropped_img)
        plt.title(f'裁剪图像 {i}')
        plt.axis('off')
        plt.savefig(f'cropped_image_{i}.png')
        plt.close()
