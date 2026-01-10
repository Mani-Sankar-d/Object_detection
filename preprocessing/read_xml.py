VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]


import xml.etree.ElementTree as ET


# In preprocessing/read_xml.py
def read_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []
    difficult = []  # Add this

    size = root.find('size')
    H = int(size.find('height').text)
    W = int(size.find('width').text)

    for obj in root.findall('object'):
        # Check if difficult
        diff = obj.find('difficult')
        is_difficult = int(diff.text) == 1 if diff is not None else False
        difficult.append(is_difficult)  # Add this

        # Get class label
        class_name = obj.find('name').text
        label = VOC_CLASSES.index(class_name)
        labels.append(label)

        # Get bounding box
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes, labels, H, W, difficult  # Add difficult to return

import numpy as np

def voc_to_target(xml_path, S=7, num_classes=20):
    boxes, labels, H, W,_ = read_voc_xml(xml_path)

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


