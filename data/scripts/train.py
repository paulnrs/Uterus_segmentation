"""Training pipeline for Detectron2 uterus segmentation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import hooks
from detectron2.utils.visualizer import Visualizer
from pycocotools.coco import COCO


class EarlyStoppingHook(hooks.HookBase):
    """Hook that stops training when a metric stops improving."""

    def __init__(
        self,
        metric: str,
        patience: int,
        *,
        maximize: bool = True,
        min_delta: float = 0.0,
        warmup_iters: int = 0,
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.maximize = maximize
        self.min_delta = min_delta
        self.warmup_iters = warmup_iters

        self._best_score: Optional[float] = None
        self._best_iter: int = -1
        self._bad_epochs: int = 0
        self._last_eval_iter: int = -1

    def after_step(self) -> None:
        storage = self.trainer.storage
        try:
            history = storage.history(self.metric)
        except KeyError:
            return

        if not history:
            return

        latest = history.latest()
        eval_iter = latest.iter
        if eval_iter is None or eval_iter == self._last_eval_iter:
            return

        self._last_eval_iter = eval_iter
        score = latest.value

        if self._best_score is None:
            self._best_score = score
            self._best_iter = eval_iter
            return

        improvement = (
            score > self._best_score + self.min_delta
            if self.maximize
            else score < self._best_score - self.min_delta
        )

        if improvement:
            self._best_score = score
            self._best_iter = eval_iter
            self._bad_epochs = 0
            return

        if eval_iter < self.warmup_iters:
            return

        self._bad_epochs += 1
        if self._bad_epochs >= self.patience:
            self.trainer.storage.put_scalar(
                "early_stopping/trigger_iter", self.trainer.iter
            )
            self.trainer.storage.put_scalar(
                "early_stopping/best_score", self._best_score
            )
            self.trainer.iter = self.trainer.max_iter


class EarlyStoppingTrainer(DefaultTrainer):
    def __init__(self, cfg, *, early_stopping: Optional[Dict] = None):
        self._early_stopping_cfg = early_stopping or {}
        super().__init__(cfg)

    def build_hooks(self):
        hooks_list = super().build_hooks()
        if self._early_stopping_cfg:
            hooks_list.append(EarlyStoppingHook(**self._early_stopping_cfg))
        return hooks_list

from evaluate import ModelEvaluator, UterusSegmentationInference


class DatasetValidator:
    """Validate and analyse a COCO dataset built for uterus ultrasound images."""

    def __init__(self, annotations_path: str, images_dir: str) -> None:
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)
        self.coco = COCO(str(self.annotations_path))

    def validate_dataset(self) -> None:
        """Run a suite of sanity checks on the dataset and display statistics."""

        print("=" * 60)
        print("VALIDATION DU DATASET")
        print("=" * 60)

        print("\n📊 Statistiques générales:")
        print(f"  • Nombre d'images: {len(self.coco.imgs)}")
        print(f"  • Nombre d'annotations: {len(self.coco.anns)}")
        categories = [cat["name"] for cat in self.coco.cats.values()]
        print(f"  • Catégories: {categories}")

        missing_images = []
        for img_info in self.coco.imgs.values():
            img_path = self.images_dir / img_info["file_name"]
            if not img_path.exists():
                missing_images.append(img_info["file_name"])

        if missing_images:
            print(f"\n⚠️ Images manquantes: {len(missing_images)}")
            for img in missing_images[:5]:
                print(f"  - {img}")
        else:
            print("\n✅ Toutes les images sont présentes")

        self._analyze_annotations()

    def _analyze_annotations(self) -> None:
        areas = []
        num_points = []

        for ann in self.coco.anns.values():
            areas.append(ann.get("area", 0))

            if "segmentation" in ann:
                for seg in ann["segmentation"]:
                    num_points.append(len(seg) // 2)

        print("\n📐 Analyse des annotations:")
        if areas:
            print(f"  • Aire moyenne: {np.mean(areas):.0f} pixels²")
            print(f"  • Aire min/max: {min(areas):.0f} / {max(areas):.0f}")
        else:
            print("  • Aucune aire disponible")

        if num_points:
            print(f"  • Points moyens par polygone: {np.mean(num_points):.0f}")
        else:
            print("  • Aucun polygone détecté")

        issues = self._check_annotation_issues()
        if issues:
            print("\n⚠️ Problèmes détectés:")
            for issue in issues[:10]:
                print(f"  - {issue}")
        else:
            print("\n✅ Aucune anomalie détectée")

    def _check_annotation_issues(self) -> list[str]:
        issues: list[str] = []

        for ann_id, ann in self.coco.anns.items():
            if "segmentation" in ann:
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

    def visualize_samples(self, num_samples: int = 4) -> None:
        img_ids = random.sample(list(self.coco.imgs.keys()), min(num_samples, len(self.coco.imgs)))

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()

        for idx, img_id in enumerate(img_ids):
            img_info = self.coco.imgs[img_id]
            img_path = self.images_dir / img_info["file_name"]

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Impossible de charger {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for ann in anns:
                mask = np.maximum(mask, self.coco.annToMask(ann))

            axes[idx].imshow(img)
            masked = np.ma.masked_where(mask == 0, mask)
            axes[idx].imshow(masked, alpha=0.5, cmap="jet")

            for ann in anns:
                bbox = ann["bbox"]
                rect = Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor="yellow", linewidth=2
                )
                axes[idx].add_patch(rect)

            axes[idx].set_title(f"ID: {img_id} - {img_info['file_name'][:30]}")
            axes[idx].axis("off")

        plt.suptitle("Échantillons du Dataset", fontsize=16)
        plt.tight_layout()
        plt.show()


class UterusSegmentationTrainer:
    """Wrapper around Detectron2's ``DefaultTrainer`` configured for this project."""

    def __init__(self, config_params: Optional[dict] = None) -> None:
        self.cfg = get_cfg()
        self.setup_config(config_params)

    def setup_config(self, params: Optional[dict] = None) -> None:
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        self.cfg.SOLVER.IMS_PER_BATCH = 16
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = 3000
        self.cfg.SOLVER.STEPS = [2000, 2500]
        self.cfg.SOLVER.GAMMA = 0.1

        self.cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
        self.cfg.INPUT.MIN_SIZE_TEST = 800
        self.cfg.INPUT.MAX_SIZE_TRAIN = 1333
        self.cfg.INPUT.MAX_SIZE_TEST = 1333
        self.cfg.INPUT.FORMAT = "RGB"

        self.cfg.TEST.AUG.ENABLED = True
        self.cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000)
        self.cfg.TEST.AUG.MAX_SIZE = 1333
        self.cfg.TEST.AUG.FLIP = True

        self.cfg.OUTPUT_DIR = "./output"
        Path(self.cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
        # Set device to CPU for macOS compatibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and torch.cuda.device_count() > 1:
            # Detectron2 gère automatiquement les GPU multiples via launch, mais on peut
            # afficher l'information pour transparence.
            print(f"Utilisation de {torch.cuda.device_count()} GPU(s)")
        self.cfg.MODEL.DEVICE = device
        
        # Evaluation cadence for early stopping
        self.cfg.TEST.EVAL_PERIOD = 200

        if params:
            for key, value in params.items():
                setattr(self.cfg, key, value)

    def register_datasets(
        self,
        train_json: str,
        train_imgs: str,
        val_json: str,
        val_imgs: str,
    ) -> None:
        for name in ("uterus_train", "uterus_val"):
            try:
                DatasetCatalog.remove(name)
            except KeyError:
                pass
            try:
                MetadataCatalog.remove(name)
            except KeyError:
                pass

        register_coco_instances("uterus_train", {}, train_json, train_imgs)
        register_coco_instances("uterus_val", {}, val_json, val_imgs)

        self.cfg.DATASETS.TRAIN = ("uterus_train",)
        self.cfg.DATASETS.TEST = ("uterus_val",)

        MetadataCatalog.get("uterus_train").set(thing_classes=["uterus"])
        MetadataCatalog.get("uterus_val").set(thing_classes=["uterus"])

        print("✅ Datasets enregistrés avec succès")

    def train(self) -> DefaultTrainer:
        print("\n" + "=" * 60)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 60)

        early_stopping_cfg = {
            "metric": "segm/AP",
            "patience": 5,
            "maximize": True,
            "min_delta": 1e-4,
            "warmup_iters": self.cfg.TEST.EVAL_PERIOD,
        }

        trainer = EarlyStoppingTrainer(
            self.cfg,
            early_stopping=early_stopping_cfg,
        )
        trainer.resume_or_load(resume=False)

        print(f"\n🚀 Entraînement sur {self.cfg.SOLVER.MAX_ITER} itérations...")
        print(f"   Learning Rate: {self.cfg.SOLVER.BASE_LR}")
        print(f"   Batch Size: {self.cfg.SOLVER.IMS_PER_BATCH}")
        print(f"   Output: {self.cfg.OUTPUT_DIR}")

        trainer.train()

        print("\n✅ Entraînement terminé!")
        return trainer


def main_training_pipeline() -> Tuple[UterusSegmentationTrainer, ModelEvaluator, dict]:
    TRAIN_ANNOTATIONS = "data/train/annotations.json"
    TRAIN_IMAGES = "data/train/images"
    VAL_ANNOTATIONS = "data/val/annotations.json"
    VAL_IMAGES = "data/val/images"

    print("\n🔍 ÉTAPE 1: Validation du dataset")
    validator = DatasetValidator(TRAIN_ANNOTATIONS, TRAIN_IMAGES)
    validator.validate_dataset()
    # validator.visualize_samples()  # Commented out for terminal environment

    print("\n⚙️ ÉTAPE 2: Configuration du modèle")
    trainer = UterusSegmentationTrainer()

    print("   Utilisation des données fournies")
    trainer.register_datasets(
        TRAIN_ANNOTATIONS,
        TRAIN_IMAGES,
        VAL_ANNOTATIONS,
        VAL_IMAGES,
    )

    print("\n🚀 ÉTAPE 3: Entraînement du modèle")
    trainer_instance = trainer.train()

    print("\n📊 ÉTAPE 4: Évaluation")
    evaluator = ModelEvaluator(trainer.cfg)
    results = evaluator.evaluate_on_validation()

    with Path("output/config.yaml").open("w", encoding="utf-8") as fh:
        fh.write(trainer.cfg.dump())

    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)
    print(f"\nModèle sauvegardé dans: {trainer.cfg.OUTPUT_DIR}")
    print("Configuration sauvegardée: output/config.yaml")

    return trainer_instance, evaluator, results


if __name__ == "__main__":
    trainer, evaluator, results = main_training_pipeline()

    print("\n🔮 Test d'inférence sur nouvelle image...")
    inference = UterusSegmentationInference(
        model_path="output/model_final.pth",
        config_path="output/config.yaml",
    )

    sample_image = Path("data/test/images/sample.jpg")
    if sample_image.exists():
        test_result = inference.segment_image(sample_image)
        print(f"Détections: {test_result['num_detections']}")
        if test_result["scores"].size:
            print(f"Score max: {test_result['scores'].max():.3f}")
    else:
        print("Aucune image 'sample.jpg' trouvée dans data/test/images")
