"""
Train a classifier on the RoI.
"""

from ultralytics import YOLO

print('---')
print('Start!')
print('---')

model = YOLO("yolov8n-cls.pt")
results = model.train(data='/mnt/wks2/boris/dev/data/classification_2/', 
                      epochs=1000, patience=1000, device=[0, 1], seed=42)

print('---')
print('---')

model = YOLO("yolov8s-cls.pt")
results = model.train(data='/mnt/wks2/boris/dev/data/classification_2/', 
                      epochs=1000, patience=1000, device=[0, 1], seed=42)

print('---')
print('---')

model = YOLO("yolov8m-cls.pt")
results = model.train(data='/mnt/wks2/boris/dev/data/classification_2/', 
                      epochs=1000, patience=1000, device=[0, 1], seed=42)

print('---')
print('---')

model = YOLO("yolov8l-cls.pt")
results = model.train(data='/mnt/wks2/boris/dev/data/classification_2/', 
                      epochs=1000, patience=1000, device=[0, 1], seed=42)

print('---')
print('---')

model = YOLO("yolov8x-cls.pt")
results = model.train(data='/mnt/wks2/boris/dev/data/classification_2/', 
                      epochs=1000, patience=1000, device=[0, 1], seed=42)

print('---')
print('End!')
print('---')
