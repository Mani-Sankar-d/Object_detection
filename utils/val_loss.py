import torch
from datasets.VOC_Dataset import VOC_dataset
from torch.utils.data import DataLoader
from utils.getloss import OD_Loss

dataset = VOC_dataset(
    guide="D:/ml_data/VOCdevkit/VOC2007/ImageSets/Main/val.txt",
    image_path="D:/ml_data/VOCdevkit/VOC2007/JPEGImages",
    annotations_path="D:/ml_data/VOCdevkit/VOC2007/Annotations"
)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
device = "cuda"
def get_val_loss(model):
    model.eval()
    total_loss = 0.0
    num_of_batches = 0
    with torch.no_grad():
        for images, labels in (loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.permute(0, 2, 3, 1)
            loss = OD_Loss(labels, outputs)
            total_loss += loss.item()
            num_of_batches += 1
        return total_loss/num_of_batches
