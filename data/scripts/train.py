"""Training pipeline for Detectron2 uterus segmentation."""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, hooks
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from tqdm import tqdm

from evaluate import UterusSegmentationInference


# ====================
# Validation Hook
# ====================
class DiceValidationHook(hooks.HookBase):
    """
    Exécute la validation toutes les `eval_period` itérations.
    Sauvegarde le modèle seulement si le Dice s'améliore.
    """
    def __init__(self, eval_period: int, val_dataset_name: str, 
                 image_root: str, score_thresh: float = 0.5):
        self.eval_period = eval_period
        self.val_dataset_name = val_dataset_name
        self.image_root = image_root
        self.score_thresh = score_thresh
        self._best_dice = 0.0

    def after_step(self):
        """Appelé après chaque itération d'entraînement."""
        next_iter = self.trainer.iter + 1
        
        # Validation périodique uniquement
        if next_iter % self.eval_period != 0:
            return

        # Passer en mode évaluation
        self.trainer.model.eval()
        
        dataset_dicts = DatasetCatalog.get(self.val_dataset_name)
        dice_scores = []

        with torch.no_grad():
            for data in tqdm(dataset_dicts, desc="Validation Dice"):
                # Construire le chemin complet
                img_path = os.path.join(self.image_root, data["file_name"])
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"⚠️ Image introuvable: {img_path}")
                    continue

                # Convertir BGR → RGB si nécessaire
                if self.trainer.cfg.INPUT.FORMAT == "RGB":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                height, width = img.shape[:2]

                # Préparer l'input
                inputs = [{
                    "image": torch.as_tensor(img.transpose(2, 0, 1).astype("float32")),
                    "height": height,
                    "width": width
                }]
                
                # Inférence
                outputs = self.trainer.model(inputs)[0]["instances"].to("cpu")

                # Masque de prédiction
                pred_mask = np.zeros((height, width), dtype=bool)
                if len(outputs) > 0:
                    high_conf = outputs.scores > self.score_thresh
                    if high_conf.sum() > 0:
                        masks = outputs.pred_masks[high_conf].numpy()
                        for m in masks:
                            pred_mask |= m

                # Masque ground truth
                true_mask = np.zeros((height, width), dtype=bool)
                for ann in data.get("annotations", []):
                    segm = ann["segmentation"]
                    
                    if isinstance(segm, dict):  # RLE
                        m = mask_util.decode(segm)
                    elif isinstance(segm, list):  # Polygones
                        m = np.zeros((height, width), dtype=np.uint8)
                        for poly in segm:
                            if len(poly) >= 6:
                                poly_np = np.array(poly).reshape(-1, 2)
                                cv2.fillPoly(m, [poly_np.astype(np.int32)], 1)
                    else:
                        continue
                    
                    true_mask |= m.astype(bool)

                # Calcul du Dice
                intersection = (pred_mask & true_mask).sum()
                union = pred_mask.sum() + true_mask.sum()
                
                if union > 0:
                    dice = (2.0 * intersection) / union
                else:
                    dice = 1.0 if pred_mask.sum() == 0 else 0.0
                
                dice_scores.append(dice)

        # Calculer la moyenne
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        
        # Logger
        self.trainer.storage.put_scalar("validation/dice_score", mean_dice)
        
        print(f"\n{'='*60}")
        print(f"Validation @ iter {next_iter}")
        print(f"  Dice Score: {mean_dice:.4f}")
        print(f"  Samples: {len(dice_scores)}")
        
        # Sauvegarde conditionnelle
        if mean_dice > self._best_dice:
            print(f"  ✅ Amélioration ! {self._best_dice:.4f} → {mean_dice:.4f}")
            self._best_dice = mean_dice
            self.trainer.checkpointer.save("model_best_dice")
        else:
            print(f"  ⏸️  Pas d'amélioration ({mean_dice:.4f} <= {self._best_dice:.4f})")
        
        print(f"{'='*60}\n")

        # Retour en mode entraînement
        self.trainer.model.train()


# ====================
# Early Stopping Hook
# ====================
class DiceEarlyStoppingHook(hooks.HookBase):
    def __init__(self, patience: int = 5):
        self.patience = patience
        self._best_dice = 0.0
        self._best_iter = -1
        self._no_improvement_count = 0

    def after_step(self):
        try:
            history = self.trainer.storage.history("validation/dice_score")
        except KeyError:
            return
        
        if not history:
            return

        dice_score = history.latest()[0]
        current_iter = self.trainer.iter

        if dice_score > self._best_dice:
            self._best_dice = dice_score
            self._best_iter = current_iter
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1
            
            if self._no_improvement_count >= self.patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING")
                print(f"Best Dice: {self._best_dice:.4f} (iter {self._best_iter})")
                print(f"Pas d'amélioration depuis {self.patience} validations")
                print(f"{'='*60}\n")
                self.trainer.checkpointer.save("model_early_stopped")
                raise StopIteration


# ====================
# Trainer avec validation Dice
# ====================
class DiceTrainer(DefaultTrainer):
    def __init__(self, cfg, val_image_root: str, eval_period: int = 200):
        self.eval_period = eval_period
        self.val_image_root = val_image_root
        super().__init__(cfg)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        
        # Validation Dice
        hooks_list.append(
            DiceValidationHook(
                eval_period=self.eval_period,
                val_dataset_name="uterus_val",
                image_root=self.val_image_root,
                score_thresh=0.5,
            )
        )
        
        # Sauvegarde périodique
        hooks_list.append(
            hooks.PeriodicCheckpointer(
                checkpointer=self.checkpointer,
                period=500,
                max_iter=self.cfg.SOLVER.MAX_ITER
            )
        )
        
        return hooks_list


# ====================
# Dataset Validator
# ====================
class DatasetValidator:
    def __init__(self, annotations_path: str, images_dir: str) -> None:
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(self.annotations_path))

    def validate_dataset(self) -> None:
        print("=" * 60)
        print("VALIDATION DU DATASET")
        print("=" * 60)
        print("\nStatistiques générales:")
        print(f" • Nombre d'images: {len(self.coco.imgs)}")
        print(f" • Nombre d'annotations: {len(self.coco.anns)}")
        categories = [cat["name"] for cat in self.coco.cats.values()]
        print(f" • Catégories: {categories}")
        
        missing_images = []
        for img_info in self.coco.imgs.values():
            img_path = self.images_dir / img_info["file_name"]
            if not img_path.exists():
                missing_images.append(img_info["file_name"])
        
        if missing_images:
            print(f"\n⚠️ Images manquantes: {len(missing_images)}")
            for img in missing_images[:5]:
                print(f" - {img}")
        else:
            print("\n✅ Toutes les images sont présentes")
        
        self._analyze_annotations()

    def _analyze_annotations(self) -> None:
        areas = []
        num_points = []
        
        for ann in self.coco.anns.values():
            areas.append(ann.get("area", 0))
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                for seg in ann["segmentation"]:
                    num_points.append(len(seg) // 2)
        
        print("\nAnalyse des annotations:")
        if areas:
            print(f" • Aire moyenne: {np.mean(areas):.0f} pixels²")
            print(f" • Aire min/max: {min(areas):.0f} / {max(areas):.0f}")
        
        if num_points:
            print(f" • Points moyens par polygone: {np.mean(num_points):.1f}")
        
        issues = self._check_annotation_issues()
        if issues:
            print(f"\n⚠️ Problèmes détectés: {len(issues)}")
            for issue in issues[:10]:
                print(f" - {issue}")
        else:
            print("\n✅ Aucune anomalie détectée")

    def _check_annotation_issues(self) -> list[str]:
        issues = []
        
        for ann_id, ann in self.coco.anns.items():
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                for idx, seg in enumerate(ann["segmentation"]):
                    if len(seg) < 6:
                        issues.append(f"Ann {ann_id}: polygone {idx} avec < 3 points")
                    if len(seg) % 2 != 0:
                        issues.append(f"Ann {ann_id}: polygone {idx} coordonnées impaires")
            
            area = ann.get("area", 0)
            if area <= 0:
                issues.append(f"Ann {ann_id}: aire invalide ({area})")
            
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                issues.append(f"Ann {ann_id}: bbox invalide")
            elif any(v < 0 for v in bbox):
                issues.append(f"Ann {ann_id}: bbox négative")
        
        return issues


# ====================
# Trainer principal
# ====================
class UterusSegmentationTrainer:
    def __init__(self, config_params: Optional[dict] = None) -> None:
        self.cfg = get_cfg()
        self.setup_config(config_params)

    def setup_config(self, params: Optional[dict] = None) -> None:
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        
        # Configuration du modèle
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.SOLVER.IMS_PER_BATCH = 16
        self.cfg.SOLVER.BASE_LR = 0.0005
        self.cfg.SOLVER.MAX_ITER = 1000
        self.cfg.SOLVER.GAMMA = 0.1
        
        # Configuration des images
        self.cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
        self.cfg.INPUT.MIN_SIZE_TEST = 800
        self.cfg.INPUT.MAX_SIZE_TRAIN = 1333
        self.cfg.INPUT.MAX_SIZE_TEST = 1333
        self.cfg.INPUT.FORMAT = "RGB"
        
        # Augmentation de test
        self.cfg.TEST.AUG.ENABLED = False
        
        # Répertoire de sortie
        self.cfg.OUTPUT_DIR = "./output"
        Path(self.cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.DEVICE = device
        
        if device == "cuda" and torch.cuda.device_count() > 1:
            print(f"🚀 Utilisation de {torch.cuda.device_count()} GPU(s)")
        
        if params:
            for key, value in params.items():
                setattr(self.cfg, key, value)

    def register_datasets(
        self, train_json: str, train_imgs: str, val_json: str, val_imgs: str
    ) -> None:
        # Nettoyer les anciens datasets
        for name in ("uterus_train", "uterus_val"):
            try:
                DatasetCatalog.remove(name)
                MetadataCatalog.remove(name)
            except KeyError:
                pass
        
        # Enregistrer les datasets
        register_coco_instances("uterus_train", {}, train_json, train_imgs)
        register_coco_instances("uterus_val", {}, val_json, val_imgs)
        
        self.cfg.DATASETS.TRAIN = ("uterus_train",)
        self.cfg.DATASETS.VAL = ("uterus_val",)
        
        # Stocker le chemin des images de validation
        self.val_image_root = val_imgs
        
        # Métadonnées
        MetadataCatalog.get("uterus_train").set(thing_classes=["uterus"])
        MetadataCatalog.get("uterus_val").set(thing_classes=["uterus"])
        
        print("✅ Datasets enregistrés avec succès")

    def train(self):
        print("\n" + "=" * 60)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)

        trainer = DiceTrainer(
            cfg=self.cfg,
            val_image_root=self.val_image_root,
            eval_period=200
        )

        trainer.resume_or_load(resume=False)

        print(f"\nParamètres d'entraînement:")
        print(f"  • Max iterations: {self.cfg.SOLVER.MAX_ITER}")
        print(f"  • Learning rate: {self.cfg.SOLVER.BASE_LR}")
        print(f"  • Batch size: {self.cfg.SOLVER.IMS_PER_BATCH}")
        print(f"  • Validation Dice: toutes les 200 iterations")
        print(f"  • Output: {self.cfg.OUTPUT_DIR}")

        try:
            trainer.train()
        except StopIteration:
            print("Entraînement arrêté par early stopping")
        
        trainer.checkpointer.save("model_final")
        print("\n✅ Entraînement terminé !")
        
        return trainer


# ====================
# Pipeline principale
# ====================
def main_training_pipeline() -> Tuple[DefaultTrainer, dict]:
    TRAIN_ANNOTATIONS = "data/train/annotations.json"
    TRAIN_IMAGES = "data/train/images"
    VAL_ANNOTATIONS = "data/val/annotations.json"
    VAL_IMAGES = "data/val/images"

    print("\n" + "=" * 60)
    print("PIPELINE D'ENTRAÎNEMENT - SEGMENTATION UTERUS")
    print("=" * 60)

    print("\nÉTAPE 1: Validation du dataset")
    validator = DatasetValidator(TRAIN_ANNOTATIONS, TRAIN_IMAGES)
    validator.validate_dataset()

    print("\nÉTAPE 2: Configuration du modèle")
    trainer = UterusSegmentationTrainer()
    trainer.register_datasets(
        TRAIN_ANNOTATIONS, TRAIN_IMAGES, VAL_ANNOTATIONS, VAL_IMAGES
    )
    
    print(f"  • Warmup iterations: {trainer.cfg.SOLVER.WARMUP_ITERS}")
    print(f"  • Warmup factor: {trainer.cfg.SOLVER.WARMUP_FACTOR}")

    print("\nÉTAPE 3: Entraînement du modèle")
    trainer_instance = trainer.train()

    # Sauvegarder la config
    config_path = Path("output/config.yaml")
    with config_path.open("w", encoding="utf-8") as fh:
        fh.write(trainer.cfg.dump())

    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS !")
    print("=" * 60)
    print(f"\n📁 Modèle sauvegardé dans: {trainer.cfg.OUTPUT_DIR}")
    print(f"📄 Configuration: {config_path}")
    
    return trainer_instance, {}


if __name__ == "__main__":
    trainer, results = main_training_pipeline()
    
    print("\n" + "=" * 60)
    print("TEST D'INFÉRENCE")
    print("=" * 60)
    
    inference = UterusSegmentationInference(
        model_path="output/model_best_dice.pth",
        config_path="output/config.yaml",
    )
    
    sample_image = Path("data/test/images/sample.jpg")
    if sample_image.exists():
        test_result = inference.segment_image(sample_image)
        print(f"✅ Détections: {test_result['num_detections']}")
        if test_result["scores"].size:
            print(f"📊 Score max: {test_result['scores'].max():.3f}")
    else:
        print("⚠️ Aucune image 'sample.jpg' trouvée dans data/test/images")