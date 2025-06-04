import cv2
import numpy as np
from ultralytics import YOLO

# YOLOモデル読み込み（yolov8xを想定）
model = YOLO("yolov8x.pt")

# 画像読み込みと推論
results = model.predict("ex3.jpg", conf=0.1)
img = results[0].orig_img

# 画像の横幅を640にリサイズ（縦横比保持）
scale = 640 / img.shape[1]
img = cv2.resize(img, (640, int(img.shape[0]*scale)))

boxes = results[0].boxes

# HSVの色範囲設定（ジャージ色の例）
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])  # 審判の黒想定

def is_color_in_range(region, lower, upper):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = cv2.countNonZero(mask) / (region.shape[0] * region.shape[1])
    return ratio > 0.3  # 30%超ならTrue

for box in boxes:
    xy1 = box.data[0][0:2] * scale
    xy2 = box.data[0][2:4] * scale
    x1, y1 = map(int, xy1)
    x2, y2 = map(int, xy2)

    # 領域切り出し
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        continue

    # 審判除外（黒が多い）
    if is_color_in_range(region, lower_black, upper_black):
        continue

    # GK除外（大きい or 縦長すぎる領域を除外）
    area = (x2 - x1) * (y2 - y1)
    aspect_ratio = (y2 - y1) / (x2 - x1 + 1e-5)
    if area > 50000 or aspect_ratio > 3:
        continue

    # チーム判別
    color = (0, 255, 0)  # 緑：判別不可
    if (is_color_in_range(region, lower_red1, upper_red1) or
        is_color_in_range(region, lower_red2, upper_red2)):
        color = (0, 0, 255)  # 赤チーム
    elif is_color_in_range(region, lower_blue, upper_blue):
        color = (255, 0, 0)  # 青チーム

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

cv2.imshow("Players ex3", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
