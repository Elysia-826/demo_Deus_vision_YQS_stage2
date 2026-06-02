from ultralytics import YOLO
model = YOLO("C:\\Users\\YQS\\Desktop\\yolo\\runs\\detect\\runs\\temp_seed\\v4\\weights\\best.onnx")
#model.predict(source=r"C:\Users\YQS\Desktop\demo_Deus_vision_YQS_stage2\detect\assets\RM_TestVideo.mp4",conf=0.1, save=True)
#print(model.names)
#print(len(model.names))
results = model(r"C:\Users\YQS\Desktop\demo_Deus_vision_YQS_stage2\test.jpg",conf=0.001)
for box in results[0].boxes:
    print(model.names[int(box.cls)], float(box.conf))