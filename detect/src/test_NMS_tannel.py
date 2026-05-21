import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(r"C:\Users\YQS\Desktop\demo_Deus_vision_YQS_stage2\best.onnx", providers=['CPUExecutionProvider'])
dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
out = sess.run(None, {"images": dummy})[0]

print("Output shape:", out.shape)  # 应为 (1, 10, 8400)

for c in range(out.shape[1]):
    vals = out[0, c, :5]
    print(f"Channel {c}: {vals}")