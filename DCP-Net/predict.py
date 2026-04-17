import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('/data/jiayaxin/proj/yolo11_ori/ultralytics-main/runs/yolo11_new/train-aitod/exp/weights/best.pt')
    model.predict(source='/data/jiayaxin/proj/Relation-DETR-main/data/aitod/images/test',
                  imgsz=640,
                  project='runs/test_aitod',
                  name='exp',
                  save=True,
                  conf=0.20,
                  iou=0.70,
                )