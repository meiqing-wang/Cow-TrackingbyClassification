from ultralytics import YOLO

print('---')
print('Start!')
print('---')

model = YOLO("yolov8n.pt")
results = model.train(data="yaml.yaml", epochs=500, imgsz=640, patience=500, device=[0, 1], seed=0)

print('---')
print('---')

model = YOLO("yolov8s.pt")
results = model.train(data="yaml.yaml", epochs=500, imgsz=640, patience=500, device=[0, 1], seed=0)

print('---')
print('---')

model = YOLO("yolov8m.pt")
results = model.train(data="yaml.yaml", epochs=500, imgsz=640, patience=500, device=[0, 1], seed=0)

print('---')
print('---')

model = YOLO("yolov8l.pt")
results = model.train(data="yaml.yaml", epochs=500, imgsz=640, patience=500, device=[0, 1], seed=0)

print('---')
print('---')

model = YOLO("yolov8x.pt")
results = model.train(data="yaml.yaml", epochs=500, imgsz=640, patience=500, device=[0, 1], seed=0)

print('---')
print('End!')
print('---')