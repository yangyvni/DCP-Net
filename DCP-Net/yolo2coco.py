#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""python coco_evaluate.py  --annotations path\instances_val2017.json  --predictions path\predictions.json"""

import argparse
import json
import os
import sys
from io import StringIO
from pathlib import Path

import numpy as np

# AI-TODv2 数据集评估指标转换工具
from aitodpycocotools.coco import COCO
from aitodpycocotools.cocoeval import COCOeval
# DOTA-v1.0 数据集评估指标转换工具
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

def evaluate_coco(pred_json, anno_json, save_path):
    """
    使用pycocotools评估COCO格式的检测结果（适配扩展的15个指标）
    
    参数:
        pred_json: 预测结果的JSON文件路径
        anno_json: COCO格式的标注文件路径
        save_path: 评估结果保存的文件路径
    
    返回:
        stats: 扩展后的15项评估结果统计
    """

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    eval_start_msg = f"\n正在评估 COCO 指标，使用 {pred_json} 和 {anno_json}..."
    print(eval_start_msg)
    
    with open(anno_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'info' not in data:
        data['info'] = {"year": 2025, "version": "1.0"}
    with open(anno_json, 'w', encoding='utf-8') as f:
        json.dump(data, f)
        
    for x in [pred_json, anno_json]:
        assert os.path.isfile(x), f"文件 {x} 不存在"
    
    anno = COCO(str(anno_json))  
    pred = anno.loadRes(str(pred_json))  
    
    eval_bbox = COCOeval(anno, pred, 'bbox')
    eval_bbox.params.maxDets = [1, 100, 300]
    eval_bbox.evaluate()
    eval_bbox.accumulate()

    output_buffer = StringIO()
    old_stdout = sys.stdout
    sys.stdout = output_buffer
    
    eval_bbox.summarize()
    
    sys.stdout = old_stdout
    
    eval_print_output = output_buffer.getvalue()
    stats = eval_bbox.stats  
    
    stats_labels = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] =',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] =',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] =',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] =',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] =',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1500 ] =',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] =',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] =',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] =',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] =',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] =',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1500 ] =',
        'Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=1500 ] =',
        'Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=1500 ] =',
        'Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=1500 ] ='
    ]
    
    print(eval_print_output)
    
    save_content = f"{eval_start_msg}\n\n"
    save_content += "=" * 80 + "\n"
    save_content += "COCO 扩展评估结果（15项指标）\n"
    save_content += "=" * 80 + "\n"

    save_content += eval_print_output + "\n"

    save_content += "=" * 80 + "\n"
    save_content += "详细指标明细（扩展15项）\n"
    save_content += "=" * 80 + "\n"
    for label, value in zip(stats_labels, stats):
        save_content += f"{label} {value:.3f}\n"
    
    # 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(save_content)
    
    print(f"\n评估结果已保存至：{save_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='评估COCO格式的目标检测结果（适配扩展15项指标）')
    parser.add_argument('--annotations', 
                        default="instances_val2017.json", 
                        type=str, help='COCO格式的标注文件路径')

    parser.add_argument('--predictions', 
                        default="predictions.json", 
                        type=str, help='预测结果的JSON文件路径')
    
    args = parser.parse_args()
    
    pred_json = Path(args.predictions).resolve()
    anno_json = Path(args.annotations).resolve()
    save_path = pred_json.parent / "save_val.txt"
    
    # 评估并保存结果
    stats = evaluate_coco(pred_json, anno_json, save_path)


if __name__ == '__main__':
    main()