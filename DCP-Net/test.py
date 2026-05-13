import os
os.environ['ULTRALYTICS_OFFLINE'] = 'True'
from ultralytics import YOLO		

model = YOLO('runs/ours/aitodv2/ours_aitodv2/weights/best.pt')

metrics = model.val(
    data='aitodv2-val.yaml', # AI-TODv2 config file path; DOTA-v1.0 config file path：dotav1.0-val.yaml
    imgsz=640,          
    batch=4,               	
    device='0',            
    save_json=True,   
    # conf=0.20,      
    project= "runs/ours/aitodv2/ours_aitodv2/val", 
    name="val_ours_aitodv2"    
)
