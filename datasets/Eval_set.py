import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from preprocessing.read_xml import voc_to_target

class Eval_set(Dataset):
    def __init__(self, guide, image_path, annotations_path):
        self.image_path = image_path
        self.annotations_path = annotations_path
        with open(guide) as f:
            self.ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.image_path, img_id + ".jpg")
        image = Image.open(img_path).convert("RGB").resize((224,224))
        image = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0

        return image, idx
