import torch
# from datasets.VOC_Dataset import Eval_set
from torch.utils.data import DataLoader
import numpy as np
from utils.IOU import IOU_eval
obj_thres = 0.5
conf_thres = 0.5
IOU_thres = 0.5

def decode_box(i, j, x,y,w,h,S=7,W=224,H=224):
    cell_w = W/S
    cell_h = H/S
    cx = (j+x)*cell_w
    cy = (i+y)*cell_h
    bw = w*W
    bh = h*H
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return torch.tensor([x1, y1, x2, y2])

def NMS(predictions, cls):
    results=[]
    B,H,W,C = predictions.shape
    for b in range(B):
        img_results = []
        preds = predictions[b].view(-1,C)
        conf = preds[:,4]*preds[:,5+cls] #H*W
        conf_mask = conf>conf_thres
        conf = conf*conf_mask
        while torch.max(conf)>conf_thres:
            idx = conf.argmax(-1)
            chosen_box = decode_box(idx//7,idx%7,*preds[idx,0:4])
            for j,box in enumerate(preds):
                if conf[j]>conf_thres and IOU_eval(chosen_box, decode_box(j//7,j%7,*box[0:4]),*preds[j,0:4])>IOU_thres:
                    conf[j]=0
            img_results.append(chosen_box)
        results.append(img_results)
    return results


def solve(predictions):
    predictions = torch.sigmoid(predictions[..., 0:4])
    mask = predictions[...,4]>obj_thres # b,7,7
    val_predictions = predictions*mask.unsqueeze(-1)
    results = []
    for cls in range(20):
        results.append(NMS(val_predictions, cls))


