import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/ours/dota/ours_dota/weights/best.pt')
    model.predict(source='examples/images/P2794_3600_2400.jpg',
                  task='detect',
                  imgsz=640,
                  project='runs/test_dota',
                  name='P2794_3600_2400',
                  save=True,
                  conf=0.20,
                  iou=0.70,
                )