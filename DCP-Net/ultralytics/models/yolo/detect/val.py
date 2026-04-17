# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import contextlib

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (List[int]): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (List[Any]): List for storing ground truth labels for hybrid saving.
        jdict (List[Dict[str, Any]]): List for storing JSON detection results.
        stats (Dict[str, List[torch.Tensor]]): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (Dict[str, Any], optional): Arguments for the validator.
            _callbacks (List[Any], optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch of images for YOLO validation.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.

        Returns:
            (Dict[str, Any]): Preprocessed batch.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in {"batch_idx", "cls", "bboxes"}:
            batch[k] = batch[k].to(self.device)

        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        # self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.class_map = list(range(0, len(model.names)))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        # print("模型类别顺序（model.names）：", self.names)  # 关键：打印模型类别顺序
        # print("模型类别数量：", self.nc)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[Dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains
                'bboxes', 'conf', 'cls', and 'extra' tensors.
        """
        # print(f" !!!!!!!!!!!!!!!!!self.args.iou:{ self.args.iou}, self.args.conf:{ self.args.conf}")
        outputs = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (Dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (Dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (Dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (Dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            batch (Dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # Save
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculate and return metrics statistics.

        Returns:
            (Dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Return correct prediction matrix.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (Dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: Optional[int] = None) -> torch.utils.data.Dataset:
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot validation image samples.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int, max_det: Optional[int] = None
    ) -> None:
        """
        Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (Optional[int]): Maximum number of detections to plot.
        """
        # TODO: optimize this
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        # TODO: fix this
        batched_preds["bboxes"][:, :4] = ops.xyxy2xywh(batched_preds["bboxes"][:, :4])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (Dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (Tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> None:
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (Dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            pbatch (Dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """
        stem = Path(pbatch["im_file"]).stem
        # image_id = int(stem) if stem.isnumeric() else stem
        image_id = str(stem) 
            # 关键修改：强制转换为整数（处理纯数字字符串，非数字则报错提示）
        # try:
        #     image_id = int(stem)  # 无论stem是字符串还是数字，直接转整数
        # except ValueError:
        #     # 若文件名前缀不是纯数字，可根据实际情况处理：要么报错，要么映射为合法整数
        #     raise ValueError(f"图片文件名前缀 '{stem}' 不是纯数字，无法转换为整数image_id，请检查文件名格式")
        
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (Dict[str, Any]): Current statistics dictionary.

        Returns:
            (Dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    # def coco_evaluate(
    #     self,
    #     stats: Dict[str, Any],
    #     pred_json: str,
    #     anno_json: str,
    #     iou_types: Union[str, List[str]] = "bbox",
    #     suffix: Union[str, List[str]] = "Box",
    # ) -> Dict[str, Any]:
    #     """
    #     Evaluate COCO/LVIS metrics using faster-coco-eval library.

    #     Performs evaluation using the faster-coco-eval library to compute mAP metrics
    #     for object detection. Updates the provided stats dictionary with computed metrics
    #     including mAP50, mAP50-95, and LVIS-specific metrics if applicable.

    #     Args:
    #         stats (Dict[str, Any]): Dictionary to store computed metrics and statistics.
    #         pred_json (str | Path]): Path to JSON file containing predictions in COCO format.
    #         anno_json (str | Path]): Path to JSON file containing ground truth annotations in COCO format.
    #         iou_types (str | List[str]]): IoU type(s) for evaluation. Can be single string or list of strings.
    #             Common values include "bbox", "segm", "keypoints". Defaults to "bbox".
    #         suffix (str | List[str]]): Suffix to append to metric names in stats dictionary. Should correspond
    #             to iou_types if multiple types provided. Defaults to "Box".

    #     Returns:
    #         (Dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
    #     """
    #     if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
    #         LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
    #         try:
    #             for x in pred_json, anno_json:
    #                 assert x.is_file(), f"{x} file not found"
    #             iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
    #             suffix = [suffix] if isinstance(suffix, str) else suffix
    #             check_requirements("faster-coco-eval>=1.6.7")
    #             from faster_coco_eval import COCO, COCOeval_faster

    #             anno = COCO(anno_json)
    #             pred = anno.loadRes(pred_json)
    #             for i, iou_type in enumerate(iou_types):
    #                 val = COCOeval_faster(
    #                     anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
    #                 )
    #                 # val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
    #                 val.evaluate()
    #                 val.accumulate()
    #                 val.summarize()

    #                 # update mAP50-95 and mAP50
    #                 stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
    #                 stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

    #                 if self.is_lvis:
    #                     stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
    #                     stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
    #                     stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

    #             if self.is_lvis:
    #                 stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
    #         except Exception as e:
    #             LOGGER.warning(f"faster-coco-eval unable to run: {e}")
    #     return stats

    # def coco_evaluate(
    #     self,
    #     stats: Dict[str, Any],
    #     pred_json: str,
    #     anno_json: str,
    #     iou_types: Union[str, List[str]] = "bbox",
    #     suffix: Union[str, List[str]] = "Box",
    # ) -> Dict[str, Any]:
    #     """
    #     评估 COCO/LVIS 指标使用 aitodpycocotools（替换 faster-coco-eval）
    #     适配字符串类型 image_id，保持 mAP 计算逻辑一致
    #     """
    #     if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
    #         LOGGER.info(f"\nEvaluating mAP using {pred_json} and {anno_json}...")
    #         try:
    #             # 验证文件存在
    #             for x in pred_json, anno_json:
    #                 assert Path(x).is_file(), f"{x} file not found"
                
    #             # 统一参数格式
    #             iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
    #             suffix = [suffix] if isinstance(suffix, str) else suffix
                
    #             # 导入 aitodpycocotools（替换 faster-coco-eval）
    #             from aitodpycocotools.coco import COCO
    #             from aitodpycocotools.cocoeval import COCOeval

    #             # 加载标注和预测文件
    #             anno = COCO(str(anno_json))  # 转为字符串路径避免兼容问题
    #             pred = anno.loadRes(str(pred_json))  # 加载预测结果

    #             # 遍历所有 IoU 类型进行评估
    #             for i, iou_type in enumerate(iou_types):
    #                 val = COCOeval(anno, pred, iouType=iou_type)
                    
    #                 # 可选：指定评估的图片 ID（与原逻辑一致，注释可保留或启用）
    #                 # val.params.imgIds = [Path(x).stem for x in self.dataloader.dataset.im_files]  # 字符串类型 imgIds
                    
    #                 # 执行评估流程
    #                 val.evaluate()
    #                 val.accumulate()
    #                 val.summarize()

    #                 # 更新 mAP 指标（与原代码字段一致，确保后续统计正常）
    #                 # COCOeval 的 stats 数组顺序：[AP@0.5:0.95, AP@0.5, AP@0.75, AP_small, AP_medium, AP_large, AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large]
    #                 stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats[1]  # AP@0.5
    #                 stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats[0]  # AP@0.5:0.95

    #                 # LVIS 专属指标（若需支持，保持原逻辑）
    #                 if self.is_lvis:
    #                     stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict.get("APr", 0.0)
    #                     stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict.get("APc", 0.0)
    #                     stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict.get("APf", 0.0)

    #             # LVIS 适配：使用 box mAP50-95 作为 fitness
    #             if self.is_lvis:
    #                 stats["fitness"] = stats["metrics/mAP50-95(B)"]

    #         except Exception as e:
    #             LOGGER.warning(f"aitodpycocotools evaluation failed: {e}")
    #     return stats
    def coco_evaluate(
        self,
        stats: Dict[str, Any],
        pred_json: Union[str, Path],
        anno_json: Union[str, Path],
        iou_types: Union[str, List[str]] = "bbox",
        suffix: Union[str, List[str]] = "Box",
    ) -> Dict[str, Any]:
        """
        使用 aitodpycocotools（与 COCOeval API 兼容）评估，并将 COCOeval 的原始打印输出
        （evaluate/accumulate/summarize 所产生的文本，格式与你截图一致）
        保存到 self.save_dir / "save_val.txt"。同时将常用数值写入 stats 以供后续使用。
        """
        # 仅在需要保存 JSON 且有预测结果时才运行评估
        if not (self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict)):
            return stats

        LOGGER.info(f"\nEvaluating mAP using {pred_json} and {anno_json}...")
        try:
            # 确保路径为 Path 并存在
            pred_json = Path(pred_json)
            anno_json = Path(anno_json)
            for x in (pred_json, anno_json):
                assert x.is_file(), f"{x} file not found"

            # 标准化参数
            iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
            suffix = [suffix] if isinstance(suffix, str) else suffix

            # 导入评估工具
            from aitodpycocotools.coco import COCO
            from aitodpycocotools.cocoeval import COCOeval

            # 加载标注与预测
            anno = COCO(str(anno_json))
            pred = anno.loadRes(str(pred_json))

            # 用于保存 COCOeval 的标准输出文本（完整 capture）
            full_capture = io.StringIO()

            # 对每个 iou_type 进行评估，并捕获标准输出（evaluate/accumulate/summarize 的 print）
            eval_results_texts = []
            for i, iou_type in enumerate(iou_types):
                val = COCOeval(anno, pred, iouType=iou_type)

                # 有些实现会打印很多信息，使用 redirect_stdout 捕获
                with contextlib.redirect_stdout(full_capture):
                    # 执行评估
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                # 从 val.stats 提取关键指标写入 stats（保持与你原代码一致）
                # COCOeval.stats: [AP (0:0.5:0.95), AP@0.5, AP@0.75, AP_small, AP_medium, AP_large,
                #                   AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large]
                try:
                    stats[f"metrics/mAP50({suffix[i][0]})"] = float(val.stats[1])   # AP@0.5
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = float(val.stats[0])  # AP@0.5:0.95
                except Exception:
                    # 若 val.stats 不存在或长度不足，设置为 0.0（或保持原值）
                    stats[f"metrics/mAP50({suffix[i][0]})"] = stats.get(f"metrics/mAP50({suffix[i][0]})", 0.0)
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = stats.get(f"metrics/mAP50-95({suffix[i][0]})", 0.0)

                # LVIS 特殊字段（保持兼容）
                if self.is_lvis:
                    # val 可能包含更丰富字典字段，尽量兼容读取
                    stats[f"metrics/APr({suffix[i][0]})"] = getattr(val, "stats_as_dict", {}).get("APr", 0.0)
                    stats[f"metrics/APc({suffix[i][0]})"] = getattr(val, "stats_as_dict", {}).get("APc", 0.0)
                    stats[f"metrics/APf({suffix[i][0]})"] = getattr(val, "stats_as_dict", {}).get("APf", 0.0)

            # 将捕获的 stdout 文本写入文件（保存为 self.save_dir / "save_val.txt"）
            captured_text = full_capture.getvalue()
            save_txt_path = Path(self.save_dir) / "save_val.txt"
            save_txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_txt_path, "w", encoding="utf-8") as f:
                # 先写入 header 信息（可选）
                f.write(f"Evaluating mAP using {pred_json} and {anno_json}...\n")
                # 写入 COCOeval 的全部原始输出
                f.write(captured_text)

            LOGGER.info(f"Validation results saved to {save_txt_path}")

            # LVIS fitness 逻辑（保留）
            if self.is_lvis:
                stats["fitness"] = stats.get("metrics/mAP50-95(B)", stats.get("metrics/mAP50-95(B)", 0.0))

        except Exception as e:
            LOGGER.warning(f"aitodpycocotools evaluation failed: {e}")
        return stats
