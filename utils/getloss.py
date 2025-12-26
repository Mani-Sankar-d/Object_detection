import torch
from utils.IOU import IOU

criterion_obj_loss =  torch.nn.BCEWithLogitsLoss()
criterion_cls_loss = torch.nn.CrossEntropyLoss()

def OD_Loss(labels,outputs):
    obj_mask = labels[..., 4] == 1
    noobj_mask = labels[..., 4] == 0

    obj_loss = criterion_obj_loss(
        outputs[..., 4][obj_mask],
        labels[..., 4][obj_mask]
    )

    noobj_loss = criterion_obj_loss(
        outputs[..., 4][noobj_mask],
        labels[..., 4][noobj_mask]
    )

    total_obj_loss = obj_loss + 0.1 * noobj_loss

    obj_mask = labels[:, :, :, 4] == 1

    pred_bbox = outputs[:, :, :, :4][obj_mask]
    label_bbox = labels[:, :, :, :4][obj_mask]
    pred_decoded = torch.sigmoid(pred_bbox[..., 0:4])
    l1_loss = torch.nn.functional.l1_loss(pred_decoded, label_bbox)
    iou_loss = 1 - IOU(pred_decoded, label_bbox).mean()
    reg_loss = l1_loss + iou_loss

    pred_cls = outputs[:, :, :, 5:][obj_mask]
    label_cls = labels[:, :, :, 5:][obj_mask].argmax(-1)
    cls_loss = criterion_cls_loss(pred_cls, label_cls)

    total_loss = reg_loss + cls_loss + total_obj_loss
    return total_loss