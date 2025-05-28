import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("ex2.jpg", conf=0.1)

# 入力画像
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes


max_area = 0
max_box = None

# 人物クラスのみを対象に処理
for box in boxes:
    cls_id = int(box.cls[0])
    if cls_id != 0:
        continue  # person以外は無視

    x1, y1, x2, y2 = box.data[0][0:4]
    area = (x2 - x1) * (y2 - y1)

    if area > max_area:
        max_area = area
        max_box = [x1, y1, x2, y2]

# 面積最大の人物にだけ赤枠を描く
if max_box:
    x1, y1, x2, y2 = map(int, max_box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)


cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
