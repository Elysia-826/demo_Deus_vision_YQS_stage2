import cv2
import numpy as np
from ultralytics import YOLO

# 加载你的 ONNX 模型
model = YOLO("C:\\Users\\YQS\\Desktop\\demo_Deus_vision_YQS_stage2\\best.onnx", task="detect")

# 随便拍一张图或从测试视频截一帧保存为 test.jpg
frame = cv2.imread("C:\\Users\\YQS\\Desktop\\demo_Deus_vision_YQS_stage2\\test.jpg")

# 推理
results = model(frame, verbose=False)[0]

# 打印检测到的框数
print(f"Detected {len(results.boxes)} objects")

# 打印每个框的坐标、置信度、类别
for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = box.conf[0].item()
    cls = int(box.cls[0])
    print(f"Box {i}: x1={x1:.1f} y1={y1:.1f} x2={x2:.1f} y2={y2:.1f} conf={conf:.2f} cls={cls}")