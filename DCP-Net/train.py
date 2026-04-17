
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO("yolo11s-dcp.yaml") 

    model.train(data='dataset/dataset.yaml', # AI-TODv2 config file path; DOTA-v1.0 config file path：dataset_dota.yaml
                cache=False, 
                imgsz=640,
                epochs=300,
                batch=8,
                val=True,
                close_mosaic=10,
                device='0',
                project='runs/ours/aitodv2',
                optimizer='SGD',
                name='ours_aitodv2',
                )
