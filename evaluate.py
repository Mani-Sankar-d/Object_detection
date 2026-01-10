import torch
from datasets.Eval_set import Eval_set
from torch.utils.data import DataLoader
import numpy as np
from utils.IOU import IOU_eval
import os
from preprocessing.read_xml import read_voc_xml
from models.model import Model
from tqdm import tqdm

dataset = Eval_set(
    guide="VOC2007/ImageSets/Layout/test.txt",
    image_path="VOC2007/JPEGImages",
    annotations_path="VOC2007/Annotations"
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

obj_thres = 0.5
conf_thres = 0.5
NMS_IOU_thres = 0.5
mAP_IOU_thres = 0.5  # Standard VOC threshold

device = "cuda"


def decode_box(grid_i, grid_j, x, y, w, h, S=7, W=224, H=224):
    """Decode box from grid cell predictions to image coordinates."""
    cx = (grid_j + x) * (W / S)
    cy = (grid_i + y) * (H / S)
    bw = w * W
    bh = h * H

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return torch.stack([x1, y1, x2, y2])


def NMS_single(predictions, S=7):
    """Apply Non-Maximum Suppression to predictions."""
    results = []
    B, H, W, C = predictions.shape

    if H != S or W != S:
        S = H

    for b in range(B):
        img_results = []
        preds = predictions[b].view(-1, C)

        obj = preds[:, 4]
        cls_probs = preds[:, 5:]
        cls_score, cls_id = cls_probs.max(dim=1)

        raw_conf = obj * cls_score
        conf = raw_conf.clone()

        while conf.max() > conf_thres:
            idx = conf.argmax()

            grid_i = idx // W
            grid_j = idx % W

            box = decode_box(grid_i, grid_j, *preds[idx, :4], S=S)

            box_clamped = box.clone()
            box_clamped[0] = box_clamped[0].clamp(0, 224)
            box_clamped[1] = box_clamped[1].clamp(0, 224)
            box_clamped[2] = box_clamped[2].clamp(0, 224)
            box_clamped[3] = box_clamped[3].clamp(0, 224)

            if box_clamped[2] <= box_clamped[0] or box_clamped[3] <= box_clamped[1]:
                conf[idx] = 0
                continue

            img_results.append((
                box_clamped,
                cls_id[idx].item(),
                raw_conf[idx].item()
            ))

            conf[idx] = 0

            for j in range(len(conf)):
                if conf[j] > 0 and cls_id[j] == cls_id[idx]:
                    other_grid_i = j // W
                    other_grid_j = j % W
                    other_box = decode_box(other_grid_i, other_grid_j, *preds[j, :4], S=S)
                    other_box_clamped = torch.clamp(other_box, 0, 224)

                    if (other_box_clamped[2] <= other_box_clamped[0] or
                            other_box_clamped[3] <= other_box_clamped[1]):
                        conf[j] = 0
                        continue

                    iou = IOU_eval(
                        box_clamped[0], box_clamped[1], box_clamped[2], box_clamped[3],
                        other_box_clamped[0], other_box_clamped[1],
                        other_box_clamped[2], other_box_clamped[3]
                    )

                    if iou > NMS_IOU_thres:
                        conf[j] = 0

        results.append(img_results)

    return results


def solve(predictions):
    """Apply sigmoid, softmax and NMS to raw predictions."""
    predictions = predictions.clone()

    shape = predictions.shape

    if len(shape) == 4:
        B, dim1, dim2, dim3 = shape

        if dim1 == 25 and dim2 == 7 and dim3 == 7:
            predictions = predictions.permute(0, 2, 3, 1)
        elif dim1 == 7 and dim2 == 25 and dim3 == 7:
            predictions = predictions.permute(0, 1, 3, 2)
        elif dim1 == 7 and dim2 == 7 and dim3 == 25:
            pass
        else:
            raise ValueError(f"Unexpected prediction shape: {shape}")

    predictions[..., 0:4] = torch.sigmoid(predictions[..., 0:4])
    predictions[..., 4] = torch.sigmoid(predictions[..., 4])
    predictions[..., 5:] = torch.softmax(predictions[..., 5:], dim=-1)

    return NMS_single(predictions)


def predictions_in_format(model):
    """Run model on dataset and return all predictions."""
    box_list = []
    model.eval()

    with torch.no_grad():
        for image, idx in tqdm(loader, desc="Generating predictions"):
            image = image.to(device)
            predictions = model(image)
            results = solve(predictions)

            for b in range(len(idx)):
                img_id = idx[b].item() if torch.is_tensor(idx[b]) else idx[b]

                for box, cls, conf in results[b]:
                    box_list.append((img_id, cls, conf, box.cpu()))

    return box_list


def get_gt(dataset, target_size=224):
    """Load ground truth boxes from dataset annotations."""
    gt = {}
    gt_difficult = {}

    for idx in tqdm(range(len(dataset)), desc="Loading ground truth"):
        img_id = dataset.ids[idx]
        xml_path = os.path.join(dataset.annotations_path, img_id + ".xml")

        # read_voc_xml now returns difficult flags
        boxes, labels, H, W, difficult = read_voc_xml(xml_path)

        sx = target_size / W
        sy = target_size / H

        image_gt = {}
        image_difficult = {}

        for (xmin, ymin, xmax, ymax), cls, is_diff in zip(boxes, labels, difficult):
            xmin_scaled = xmin * sx
            xmax_scaled = xmax * sx
            ymin_scaled = ymin * sy
            ymax_scaled = ymax * sy

            if cls not in image_gt:
                image_gt[cls] = []
                image_difficult[cls] = []

            image_gt[cls].append(torch.tensor([xmin_scaled, ymin_scaled,
                                               xmax_scaled, ymax_scaled]))
            image_difficult[cls].append(is_diff)

        gt[idx] = image_gt
        gt_difficult[idx] = image_difficult

    return gt, gt_difficult


def compute_ap(recalls, precisions):
    """Compute Average Precision using the precision-recall curve."""
    recalls = np.array(recalls)
    precisions = np.array(precisions)

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def compute_mAP(predictions, gt, gt_difficult, num_classes=20, iou_thres=0.5):
    """Compute mean Average Precision (mAP) across all classes following VOC protocol."""
    APs = []

    for cls in range(num_classes):
        cls_preds = [p for p in predictions if p[1] == cls]
        cls_preds.sort(key=lambda x: x[2], reverse=True)

        # Count total non-difficult GT boxes
        total_gt = 0
        gt_used = {}

        for img_id in gt:
            if cls in gt[img_id]:
                # Only count non-difficult GT boxes
                non_difficult = [not d for d in gt_difficult[img_id][cls]]
                total_gt += sum(non_difficult)
                gt_used[img_id] = [False] * len(gt[img_id][cls])

        if total_gt == 0:
            continue

        TP = np.zeros(len(cls_preds))
        FP = np.zeros(len(cls_preds))

        for i, (img_id, _, conf, pred_box) in enumerate(cls_preds):
            if img_id not in gt or cls not in gt[img_id]:
                FP[i] = 1
                continue

            gt_boxes = gt[img_id][cls]
            difficult_flags = gt_difficult[img_id][cls]
            best_iou = 0
            best_j = -1

            # Find best matching GT box
            for j, gt_box in enumerate(gt_boxes):
                iou = IOU_eval(
                    pred_box[0], pred_box[1], pred_box[2], pred_box[3],
                    gt_box[0], gt_box[1], gt_box[2], gt_box[3]
                )
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            # VOC protocol: ignore matches to difficult objects
            if best_iou >= iou_thres:
                if difficult_flags[best_j]:
                    # Matched a difficult object - ignore (don't count as TP or FP)
                    continue
                elif not gt_used[img_id][best_j]:
                    # Matched non-difficult, unused GT box
                    TP[i] = 1
                    gt_used[img_id][best_j] = True
                else:
                    # Matched already used GT box
                    FP[i] = 1
            else:
                # No match or IOU too low
                FP[i] = 1

        TP_cum = np.cumsum(TP)
        FP_cum = np.cumsum(FP)

        recalls = TP_cum / (total_gt + 1e-6)
        precisions = TP_cum / (TP_cum + FP_cum + 1e-6)

        AP = compute_ap(recalls, precisions)
        APs.append(AP)

        print(
            f"Class {cls}: AP = {AP:.4f} (GT: {total_gt}, Pred: {len(cls_preds)}, TP: {int(TP.sum())}, FP: {int(FP.sum())})")

    if len(APs) == 0:
        return 0.0

    return sum(APs) / len(APs)


# Main evaluation
if __name__ == "__main__":
    print("=" * 60)
    print("Object Detection Evaluation (VOC Protocol)")
    print("=" * 60)

    model = Model().to(device)
    ckpt = torch.load("Object_detection.best_checkpoint_1.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    predictions = predictions_in_format(model)
    gt, gt_difficult = get_gt(dataset)

    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Total GT images: {len(gt)}")

    print("\n" + "=" * 60)
    print("Computing mAP (ignoring difficult objects)...")
    print("=" * 60)

    mAP = compute_mAP(predictions, gt, gt_difficult, iou_thres=mAP_IOU_thres)

    print("\n" + "=" * 60)
    print(f"Final Result: mAP@{mAP_IOU_thres} = {mAP:.4f}")
    print("=" * 60)