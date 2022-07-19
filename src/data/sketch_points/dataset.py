import os
import pickle as pkl
from typing import Any

import numpy as np
from common.utils import rgb2gray
from omegaconf import ValueNode, DictConfig
from skimage import io
from torch.utils.data import Dataset


class DataSet(Dataset):
    def __init__(self, name: ValueNode, path_sketch: str, path_pt: str, path_camera: str,
                 path_density: str, file_list: str, cfg: DictConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.name = name

        self.path_sketch = path_sketch
        self.path_pt = path_pt
        self.path_camera = path_camera
        self.path_density = path_density
        self.pkl_list = []
        with open(file_list, 'r') as f:
            while (True):
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)

    def __len__(self) -> int:
        return len(self.pkl_list)

    def __getitem__(
            self, idx
    ) -> Any:
        camera_path = os.path.join(self.path_camera, self.pkl_list[idx])
        density_path = os.path.join(self.path_density, self.pkl_list[idx])
        pointcloud_path = os.path.join(self.path_pt, "/".join(self.pkl_list[idx].split('/')[:2] + ["pt.dat"]))
        raw_pointcloud = pkl.load(open(pointcloud_path, 'rb'))
        cam_mat, cam_pos = pkl.load(open(camera_path, 'rb'))
        density_map = pkl.load(open(density_path, 'rb')).astype('float32')
        density_map /= np.sum(density_map)
        density_map = density_map[None, ...]
        gt_pointcloud = np.dot(raw_pointcloud - cam_pos, cam_mat.transpose())
        gt_pointcloud[:, 2] -= np.mean(gt_pointcloud[:, 2])
        sketch_path = os.path.join(self.path_sketch, self.pkl_list[idx].replace('.dat', '.png'))
        sketch = io.imread(sketch_path)
        sketch[np.where(sketch[:, :, 3] == 0)] = 255
        sketch = sketch.astype('float32') / 255
        sketch = ((np.transpose(rgb2gray(sketch[:, :, :3]), (2, 0, 1)) - .5) * 2).astype('float32')
        metadata = self.pkl_list[idx][:-4]
        item = (sketch, gt_pointcloud, density_map, metadata)
        return item

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}"
