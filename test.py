import torch
import numpy as np
import cv2
from PIL import Image

from models.model import Model
from utils.IOU import IOU

# ---------------- CONFIG ----------------
IMAGE_PATH = "D:/ml_data/VOCdevkit/VOC2007/JPEGImages/000079.jpg"
CHECKPOINT = "Object_detection.best_checkpoint.pth"

S = 7
INPUT_SIZE = 224
CONF_THRESH = 0.6
IOU_THRESH = 0.5

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


# ---------- Load model ----------
model = Model().to(device)
ckpt = torch.load("Object_detection.best_checkpoint.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# ---------- Load image ----------
img_pil = Image.open(IMAGE_PATH).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
img_np = np.array(img_pil)

img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(device)


# ---------- Forward ----------
with torch.no_grad():
    outputs = model(img_tensor)
    outputs = outputs.permute(0, 2, 3, 1)[0]   # (S,S,25)

    # decode
    outputs[..., 0:4] = torch.sigmoid(outputs[..., 0:4])
    outputs[..., 4]   = torch.sigmoid(outputs[..., 4])
    outputs[..., 5:]  = torch.softmax(outputs[..., 5:], dim=-1)


# ---------- Collect predictions ----------
preds = []

for i in range(S):
    for j in range(S):
        obj = outputs[i, j, 4].item()
        if obj < CONF_THRESH:
            continue

        cls = torch.argmax(outputs[i, j, 5:]).item()
        cls_conf = outputs[i, j, 5 + cls].item()
        score = obj * cls_conf

        if score < CONF_THRESH:
            continue

        x, y, w, h = outputs[i, j, 0:4]

        # decode to pixel space (224x224)
        cx = (j + x.item()) * (INPUT_SIZE / S)
        cy = (i + y.item()) * (INPUT_SIZE / S)
        bw = w.item() * INPUT_SIZE
        bh = h.item() * INPUT_SIZE

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        preds.append([x1, y1, x2, y2, score, cls])


# ---------- Simple NMS ----------
final = []

preds = sorted(preds, key=lambda x: x[4], reverse=True)

while preds:
    best = preds.pop(0)
    final.append(best)

    preds = [
        p for p in preds
        if IOU(
            torch.tensor(best[:4]).unsqueeze(0),
            torch.tensor(p[:4]).unsqueeze(0)
        ) < IOU_THRESH
    ]


# ---------- Draw ----------
img_draw = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

for x1, y1, x2, y2, score, cls in final:
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0,255,0), 2)

    label = f"{VOC_CLASSES[cls]} {score:.2f}"
    cv2.putText(
        img_draw,
        label,
        (x1, max(10, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1,
        cv2.LINE_AA
    )

cv2.imshow("detections", img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.destroyAllWindows()
