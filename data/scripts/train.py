"""
Training pipeline for Detectron2 uterus segmentation
VALIDATION ONLINE = IDENTIQUE VALIDATION OFFLINE
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, hooks, DefaultPredictor
from pycocotools import mask as mask_util
from pycocotools.coco import COCO


# ======================================================
# VALIDATION DICE HOOK (ALIGNÉ OFFLINE)
# ======================================================
class DiceValidationHook(hooks.HookBase):
    def __init__(self, eval_period, val_dataset_name, image_root, score_thresh=0.5):
        """
        eval_period: fréquence (en itérations) pour calculer le Dice
        val_dataset_name: nom du dataset enregistré (ex: 'uterus_val')
        image_root: dossier contenant les images
        score_thresh: seuil pour filtrer les prédictions
        """
        self.eval_period = eval_period
        self.val_dataset_name = val_dataset_name
        self.image_root = image_root
        self.score_thresh = score_thresh
        self._best_dice = 0.0

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter != 1 and next_iter % self.eval_period != 0:
            return  # On n'évalue pas à cette itération

        print(f"\n{'='*60}")
        print(f"Validation Dice @ iter {next_iter}")

        model = self.trainer.model
        model.eval()  # passage en mode évaluation

        try:
            dataset_dicts = DatasetCatalog.get(self.val_dataset_name)
        except KeyError:
            print(f"Erreur : Dataset '{self.val_dataset_name}' non trouvé.")
            return

        dice_scores = []

        with torch.no_grad():
            for data in dataset_dicts:
                img_path = os.path.join(self.image_root, data["file_name"])
                img = cv2.imread(img_path)

                if img is None:
                    print(f"⚠️ Image introuvable, ignorée : {img_path}")
                    continue  # Ignore cette image

                height, width = img.shape[:2]

                inputs = [{
                    "image": torch.as_tensor(img.astype("float32")).permute(2, 0, 1),
                    "height": height,
                    "width": width,
                }]

                outputs = model(inputs)[0]["instances"].to("cpu")

                # --- prédiction ---
                pred_mask = np.zeros((height, width), dtype=bool)
                if len(outputs) > 0:
                    high_conf = outputs.scores > self.score_thresh
                    if high_conf.any():
                        for m in outputs.pred_masks[high_conf].numpy():
                            pred_mask |= m

                # --- vérité terrain ---
                true_mask = np.zeros((height, width), dtype=bool)
                for ann in data["annotations"]:
                    segm = ann["segmentation"]
                    if isinstance(segm, dict):  # RLE
                        m = mask_util.decode(segm)
                    else:  # Polygones
                        m = np.zeros((height, width), dtype=np.uint8)
                        for poly in segm:
                            poly_np = np.array(poly).reshape(-1, 2)
                            cv2.fillPoly(m, [poly_np.astype(np.int32)], 1)
                    true_mask |= m.astype(bool)

                # --- calcul du Dice ---
                intersection = (pred_mask & true_mask).sum()
                union = pred_mask.sum() + true_mask.sum()
                dice = (2 * intersection) / union if union > 0 else 1.0
                dice_scores.append(dice)

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        self.trainer.storage.put_scalar("validation/dice_score", mean_dice)

        print(f"Dice moyen sur {len(dice_scores)} images : {mean_dice:.4f}")

        if mean_dice > self._best_dice:
            print(f"✅ Nouveau meilleur Dice ({self._best_dice:.4f} → {mean_dice:.4f})")
            self._best_dice = mean_dice
            self.trainer.checkpointer.save("model_best_dice")
        else:
            print("⏸️ Pas d'amélioration")

        print(f"{'='*60}\n")
        model.train()  # remettre en mode training




# ======================================================
# TRAINER PERSONNALISÉ
# ======================================================
class DiceTrainer(DefaultTrainer):
    def __init__(self, cfg, val_image_root, eval_period=200):
        self.val_image_root = val_image_root
        self.eval_period = eval_period
        super().__init__(cfg)

    def build_hooks(self):
        hooks_list = super().build_hooks()

        hooks_list.append(
            DiceValidationHook(
                eval_period=self.eval_period,
                val_dataset_name="uterus_val",
                image_root=self.val_image_root,
                score_thresh=0.5,
            )
        )

        return hooks_list


# ======================================================
# DATASET VALIDATOR (OPTIONNEL MAIS UTILE)
# ======================================================
class DatasetValidator:
    def __init__(self, annotations_path, images_dir):
        self.coco = COCO(annotations_path)
        self.images_dir = Path(images_dir)

    def validate_dataset(self):
        print("\nValidation du dataset...")
        missing = 0
        for img in self.coco.imgs.values():
            if not (self.images_dir / img["file_name"]).exists():
                missing += 1
        print(f"Images manquantes: {missing}")


# ======================================================
# TRAINING PIPELINE
# ======================================================
class UterusSegmentationTrainer:
    def __init__(self):
        self.cfg = get_cfg()
        self._setup_cfg()

    def _setup_cfg(self):
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.SOLVER.IMS_PER_BATCH = 16
        self.cfg.SOLVER.BASE_LR = 5e-4
        self.cfg.SOLVER.MAX_ITER = 1000

        self.cfg.INPUT.FORMAT = "BGR"
        self.cfg.OUTPUT_DIR = "./output"
        Path(self.cfg.OUTPUT_DIR).mkdir(exist_ok=True)

        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def register_datasets(self, train_json, train_imgs, val_json, val_imgs=""):
        for name in ("uterus_train", "uterus_val"):
            try:
                DatasetCatalog.remove(name)
                MetadataCatalog.remove(name)
            except KeyError:
                pass

        register_coco_instances("uterus_train", {}, train_json, train_imgs)
        register_coco_instances("uterus_val", {}, val_json, val_imgs)

        self.cfg.DATASETS.TRAIN = ("uterus_train",)
        self.cfg.DATASETS.TEST = ()

        self.val_image_root = val_imgs

        MetadataCatalog.get("uterus_train").set(thing_classes=["uterus"])
        MetadataCatalog.get("uterus_val").set(thing_classes=["uterus"])

    def train(self):
        trainer = DiceTrainer(
            self.cfg,
            val_image_root=self.val_image_root,
            eval_period=200,
        )
        trainer.resume_or_load(resume=False)
        trainer.train()
        trainer.checkpointer.save("model_final")
        return trainer


# ======================================================
# MAIN
# ======================================================
def main_training_pipeline() -> Tuple[DefaultTrainer, dict]:
    TRAIN_JSON = "data/train/annotations.json"
    TRAIN_IMAGES = "data/train/images"
    VAL_JSON = "data/val/annotations.json"
    #VAL_IMAGES = "data/val/images"

    DatasetValidator(TRAIN_JSON, TRAIN_IMAGES).validate_dataset()

    trainer = UterusSegmentationTrainer()
    trainer.register_datasets(
        TRAIN_JSON, TRAIN_IMAGES, VAL_JSON
    )

    trainer_instance = trainer.train()

    with open("output/config.yaml", "w") as f:
        f.write(trainer.cfg.dump())

    return trainer_instance, {}


if __name__ == "__main__":
    main_training_pipeline()
