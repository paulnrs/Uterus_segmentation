"""Evaluation and inference helpers for uterus segmentation models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer


class ModelEvaluator:
    """Wrapper to evaluate Detectron2 models on the validation dataset."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg.clone()
        self.predictor = DefaultPredictor(self.cfg)

    def evaluate_on_validation(self) -> Dict[str, Dict[str, float]]:
        print("\n" + "=" * 60)
        print("ÉVALUATION DU MODÈLE")
        print("=" * 60)

        evaluator = COCOEvaluator("uterus_val", self.cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, "uterus_val")

        results = inference_on_dataset(self.predictor.model, val_loader, evaluator)

        if "segm" in results:
            segm = results["segm"]
            print("\n Résultats de segmentation:")
            print(f"   • AP (IoU=0.50:0.95): {segm['AP']:.2f}")
            print(f"   • AP50: {segm['AP50']:.2f}")
            print(f"   • AP75: {segm['AP75']:.2f}")

        if "bbox" in results:
            bbox = results["bbox"]
            print("\n Résultats de détection (bbox):")
            print(f"   • AP (IoU=0.50:0.95): {bbox['AP']:.2f}")
            print(f"   • AP50: {bbox['AP50']:.2f}")
            print(f"   • AP75: {bbox['AP75']:.2f}")

        return results

    def visualize_predictions(self, image_path: Path, save_path: Optional[Path] = None):
        img = cv2.imread(str(image_path))
        outputs = self.predictor(img)

        v = Visualizer(
            img[:, :, ::-1],
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            scale=1.2,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        result_img = out.get_image()[:, :, ::-1]

        if save_path:
            cv2.imwrite(str(save_path), result_img)
            print(f" Prédiction sauvegardée: {save_path}")

        return result_img

    @staticmethod
    def calculate_dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        return 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-7)


class UterusSegmentationInference:
    """Utility class to run inference with a trained Detectron2 model."""

    def __init__(self, model_path: str, config_path: Optional[str] = None) -> None:
        self.cfg = get_cfg()

        if config_path:
            self.cfg.merge_from_file(config_path)
        else:
            self.cfg.merge_from_file(
                model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            )
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        self.predictor = DefaultPredictor(self.cfg)

    def segment_image(self, image_path: Path) -> Dict[str, np.ndarray]:
        img = cv2.imread(str(image_path))
        outputs = self.predictor(img)

        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else np.empty((0,))
        scores = instances.scores.numpy() if instances.has("scores") else np.empty((0,))

        return {
            "masks": masks,
            "scores": scores,
            "num_detections": len(masks),
        }

    def process_batch(
        self,
        image_paths: List[Path],
        *,
        save_results: bool = True,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, object]]:
        results = []
        output_dir = output_dir or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            result = self.segment_image(img_path)
            result["image_path"] = str(img_path)
            results.append(result)

            if save_results and result["num_detections"] > 0:
                mask = result["masks"][0]
                mask_path = output_dir / f"{Path(img_path).stem}_mask.png"
                cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)

        return results
