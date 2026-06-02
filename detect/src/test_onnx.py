import onnxruntime as ort

session = ort.InferenceSession(
    r"C:\Users\YQS\Desktop\yolo\runs\detect\runs\temp_seed\v3\weights\best.onnx"
)

print("INPUT:")
for i in session.get_inputs():
    print(i.name, i.shape)

print("OUTPUT:")
for o in session.get_outputs():
    print(o.name, o.shape)