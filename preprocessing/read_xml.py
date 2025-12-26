VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]


import xml.etree.ElementTree as ET

def read_voc_xml(xml_path):

    tree = ET.parse(xml_path)

    root = tree.getroot()

    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)

    boxes = []
    labels = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in VOC_CLASSES:
            continue

        cls_id = VOC_CLASSES.index(cls_name)
        bb = obj.find("bndbox")

        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls_id)

    return boxes, labels, H, W

import numpy as np

def voc_to_target(xml_path, S=7, num_classes=20):
    boxes, labels, H, W = read_voc_xml(xml_path)

    target = np.zeros((S, S, 5 + num_classes), dtype=np.float32)

    for (xmin, ymin, xmax, ymax), cls in zip(boxes, labels):

        # box center & size (pixels)
        bx = (xmin + xmax) / 2.0
        by = (ymin + ymax) / 2.0
        bw = xmax - xmin
        bh = ymax - ymin

        # grid cell
        j = int(bx / W * S)   # x → column
        i = int(by / H * S)   # y → row

        if i >= S or j >= S:
            continue

        # relative to cell
        x = (bx / W * S) - j
        y = (by / H * S) - i
        w = bw / W
        h = bh / H

        # YOLOv1: only one object per cell
        if target[i, j, 4] == 1:
            continue

        target[i, j, 0:4] = [x, y, w, h]
        target[i, j, 4] = 1.0           # objectness
        target[i, j, 5 + cls] = 1.0     # one-hot class

    return target


