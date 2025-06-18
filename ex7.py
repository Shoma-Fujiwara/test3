from ultralytics import YOLO

model = YOLO("othello.pt")

# 物体検出を実行
results = model.predict("ex4.jpg", conf=0.99)

labels = results[0].boxes.cls.tolist()

# 白と黒の石をカウント
white_count = labels.count(0)
black_count = labels.count(1)

# 結果を出力
print(f"白の石の数: {white_count}個")
print(f"黒の石の数: {black_count}個")
