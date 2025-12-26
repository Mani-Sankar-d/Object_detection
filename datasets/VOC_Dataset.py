import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from preprocessing.read_xml import voc_to_target
import random

class VOC_dataset(Dataset):
    def __init__(self, guide, image_path, annotations_path):
        super().__init__()

        self.image_path = image_path
        self.annotations_path = annotations_path
        with open(guide) as f:
            self.ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.image_path, img_id + ".jpg")
        ann_path = os.path.join(self.annotations_path, img_id + ".xml")

        image = Image.open(img_path).convert("RGB").resize((224, 224))
        label = voc_to_target(xml_path=ann_path)  # shape (S, S, 25)

        label = torch.tensor(label, dtype=torch.float32)

        # ---------- RANDOM HORIZONTAL FLIP ----------
        if random.random() < 0.5:
            # flip image
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            S = label.shape[0]
            new_label = torch.zeros_like(label)

            for i in range(S):
                for j in range(S):
                    if label[i, j, 4] == 1:  # object present
                        x, y, w, h = label[i, j, 0:4]
                        cls = label[i, j, 5:]

                        new_j = S - 1 - j
                        new_x = 1 - x

                        new_label[i, new_j, 0:4] = torch.tensor([new_x, y, w, h])
                        new_label[i, new_j, 4] = 1
                        new_label[i, new_j, 5:] = cls
            label = new_label
        # -------------------------------------------

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return image, label

