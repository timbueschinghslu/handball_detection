from ultralytics import YOLO
import torch

model = YOLO('models/best_2_datasets.pt')



results = model.predict('input_videos/test_clip_3.mp4', save=True, device='mps')
print(results[0])

print('=======================')
for box in results[0].boxes:
    print(box)