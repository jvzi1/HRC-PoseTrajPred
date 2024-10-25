import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    def __init__(self, dataset_path, clip_len, transform=None):
        self.dataset_path = dataset_path
        self.clip_len = clip_len
        self.transform = transform if transform else transforms.ToTensor()
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for label in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label)
            if os.path.isdir(label_path):
                for video_folder in os.listdir(label_path):
                    video_path = os.path.join(label_path, video_folder)
                    if os.path.isdir(video_path):
                        frames = sorted(os.listdir(video_path))
                        if len(frames) >= self.clip_len:
                            data.append((video_path, frames, int(label)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_path, frames, label = self.data[index]
        clip = []
        start_frame = 0
        for i in range(self.clip_len):
            frame_path = os.path.join(video_path, frames[start_frame + i])
            frame = Image.open(frame_path)
            clip.append(self.transform(frame))

        clip = torch.stack(clip, dim=0)  # 将多个帧堆叠为一个Tensor
        return clip, label
