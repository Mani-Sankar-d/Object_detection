import torch
import torch.nn as nn
from datasets.VOC_Dataset import VOC_dataset
from models.model import Model
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from utils.IOU import IOU
from utils.getloss import OD_Loss
from utils.checkpoints import save_checkpoint,load_checkpoint,get_best_val_loss
from utils.val_loss import get_val_loss
import os

def train(start_epoch,total_epochs,best_val_loss=float('inf')):
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0.0
        num_of_batches = 0
        for images,labels in tqdm(loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.permute(0,2,3,1)
            total_loss = OD_Loss(labels,outputs)
            total_loss.backward()
            optimizer.step()
            train_loss+=total_loss.item()
            num_of_batches += 1
        print(f"{epoch} train loss/batch is {train_loss/num_of_batches}")
        val_loss = get_val_loss(model)
        print(f"{epoch} val loss/batch is {val_loss}")
        save_checkpoint(model,optimizer,epoch,val_loss,"last_checkpoint")
        if val_loss<best_val_loss:
            save_checkpoint(model,optimizer,epoch,val_loss,"best_checkpoint")
            best_val_loss = val_loss



image_path, annotations_path = "VOC2007/JPEGImages","VOC2007/Annotations"

dataset = VOC_dataset(
    guide="VOC2007/ImageSets/Main/train.txt",
    image_path="VOC2007/JPEGImages",
    annotations_path="VOC2007/Annotations"
)

loader = DataLoader(dataset,batch_size=4, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model()
model = model.to(device)
optimizer = Adam(model.parameters(), lr=1e-4,weight_decay=1e-4)

model,_,last_epoch,best_val_loss = load_checkpoint("best_checkpoint",model,optimizer)
for p in model.backbone.parameters():
    p.requires_grad = True
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
train(last_epoch+1,50,best_val_loss)

