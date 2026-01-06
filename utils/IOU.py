import torch


def IOU(pred_bbox, label_bbox):
    x_pred, y_pred, w_pred, h_pred = pred_bbox[:,0],pred_bbox[:,1],pred_bbox[:,2],pred_bbox[:,3],
    x, y, w, h = label_bbox[:,0],label_bbox[:,1],label_bbox[:,2],label_bbox[:,3]
    x1 = torch.max(x_pred - w_pred / 2.0, x - w / 2.0)
    y1 = torch.max(y_pred - h_pred / 2.0, y - h / 2.0)

    x2 = torch.min(x_pred + w_pred / 2.0, x + w / 2.0)
    y2 = torch.min(y_pred + h_pred / 2.0, y + h / 2.0)
    width = torch.clamp(x2 - x1, 0)
    height = torch.clamp(y2 - y1, 0)
    inter = width*height
    union = w_pred*h_pred + w*h - inter
    return inter/(union+1e-6)

def IOU_eval(X1,Y1,X2,Y2, x1,y1,x2,y2):
    x1_max = torch.max(X1,x1)
    y1_max = torch.max(Y1,y1)
    x2_min = torch.min(X2,x2)
    y2_min = torch.min(Y2,y2)
    inter_w = torch.clamp(x2_min - x1_max, min=0)
    inter_h = torch.clamp(y2_min - y1_max, min=0)
    inter = inter_w * inter_h
    area1 = (X2 - X1) * (Y2 - Y1)
    area2 = (x2 - x1) * (y2 - y1)
    union = area1 + area2 - inter

    return inter/(union+1e-6)
