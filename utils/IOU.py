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

