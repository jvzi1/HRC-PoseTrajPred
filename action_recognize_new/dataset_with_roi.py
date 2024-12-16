import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VideoDatasetWithROI(Dataset):
    def __init__(self, dataset_path, images_path, clip_len, resize_height=144, resize_width=144, crop_size=112, normalize_means=(90.0, 98.0, 102.0)):
        self.dataset_path = dataset_path
        self.split = images_path
        self.clip_len = clip_len
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size
        self.normalize_means = normalize_means

        folder = os.path.join(self.dataset_path, images_path)
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
        print('Number of {} videos: {:d}'.format(images_path, len(self.fnames)))

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer, roi_mask = self.load_frames_with_roi(self.fnames[index])
        if buffer.shape[0] < self.clip_len:
            repeat_count = self.clip_len - buffer.shape[0]
            buffer = np.concatenate([buffer, buffer[-1:].repeat(repeat_count, axis=0)], axis=0)
            roi_mask = np.concatenate([roi_mask, roi_mask[-1:].repeat(repeat_count, axis=0)], axis=0)

        buffer = self.crop(buffer)
        roi_mask = self.crop(roi_mask)
        buffer = self.normalize(buffer)
        buffer = self.add_roi_as_channel(buffer, roi_mask)
        buffer = self.to_tensor(buffer)

        labels = np.array(self.label_array[index])
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def load_frames_with_roi(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        roi_mask = np.empty((frame_count, self.resize_height, self.resize_width, 1), np.dtype('float32'))

        for i, frame_name in enumerate(frames):
            image = cv2.imread(frame_name, cv2.IMREAD_UNCHANGED)
            if image is None or image.shape[2] < 4:
                raise ValueError(f"Invalid PNG with insufficient channels: {frame_name}")

            buffer[i] = image[:, :, :3].astype(np.float64)  # 前三通道为 RGB
            roi_mask[i] = image[:, :, 3:4].astype(np.float32) / 255.0  # 第四通道为 ROI

        return buffer, roi_mask

    def resize_with_padding(self, buffer, target_size):
        padded_frames = []
        for frame in buffer:
            h, w, _ = frame.shape
            scale = min(target_size / h, target_size / w)
            new_h, new_w = int(h * scale), int(w * scale)

            resized_frame = cv2.resize(frame, (new_w, new_h))
            padded_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            padded_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
            padded_frames.append(padded_frame)

        return np.array(padded_frames)

    def crop(self, buffer):
        time_index = np.random.randint(buffer.shape[0] - self.clip_len)
        height_index = np.random.randint(buffer.shape[1] - self.crop_size)
        width_index = np.random.randint(buffer.shape[2] - self.crop_size)

        buffer = buffer[time_index:time_index + self.clip_len,
                        height_index:height_index + self.crop_size,
                        width_index:width_index + self.crop_size, :]
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[self.normalize_means[0], self.normalize_means[1], self.normalize_means[2]]]])
            buffer[i] = frame
        return buffer

    def add_roi_as_channel(self, buffer, roi_mask):
        return np.concatenate([buffer, roi_mask], axis=-1)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

if __name__ == "__main__":
    train_data = VideoDatasetWithROI(dataset_path='data/ucf101', images_path='train', clip_len=16)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    val_data = VideoDatasetWithROI(dataset_path='data/ucf101', images_path='val', clip_len=16)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True, num_workers=0)

    test_data = VideoDatasetWithROI(dataset_path='data/ucf101', images_path='test', clip_len=16)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)
