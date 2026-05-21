import onnxruntime as ort
import numpy as np
import cv2

sess = ort.InferenceSession(r"C:\Users\YQS\Desktop\demo_Deus_vision_YQS_stage2\best.onnx", providers=['CPUExecutionProvider'])

# 读取一张真实测试图片（随便找一张装甲板的图）
frame = cv2.imread(r"C:\Users\YQS\Desktop\demo_Deus_vision_YQS_stage2\test.jpg")
frame = cv2.resize(frame, (640, 640))
blob = frame.astype(np.float32) / 255.0
blob = blob[:, :, ::-1].transpose(2, 0, 1)
blob = np.expand_dims(blob, axis=0)

out = sess.run(None, {"images": blob})[0]

print("Output shape:", out.shape)

# 对每个通道，打印其前 20 个值中的 min, max, mean
for c in range(out.shape[1]):
    vals = out[0, c, :]
    vmin = np.min(vals)
    vmax = np.max(vals)
    vmean = np.mean(vals)
    print(f"Channel {c}: min={vmin:.6f} max={vmax:.6f} mean={vmean:.6f}")