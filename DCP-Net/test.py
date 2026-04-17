import os
os.environ['ULTRALYTICS_OFFLINE'] = 'True'
from ultralytics import YOLO		

model = YOLO('/data/jiayaxin/proj/DCP-Net/best_weight_aitodv2/best.pt')
# model = YOLO('runs/ours/aitodv2/ours_aitodv2/weights/best.pt')
metrics = model.val(
    data='dataset/dataset-val.yaml', # AI-TODv2 config file path; DOTA-v1.0 config file path：dataset_dota-val.yaml
    imgsz=640,          
    batch=4,               	
    device='0',            
    save_json=True,         
    project= "runs/ours/aitodv2/ours_aitodv2/val", 
    name="val_ours_aitodv2"    
)
