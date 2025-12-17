"""Training pipeline for Detectron2 uterus segmentation - Version Minimale."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.comm import is_main_process

from pycocotools.coco import COCO


# ====================
# VALIDATOR (minimal)
# ====================
class DatasetValidator:
    """Validation simple du dataset."""
    
    def __init__(self, annotations_path: str, images_dir: str):
        self.coco = COCO(annotations_path)
        self.images_dir = Path(images_dir)

    def validate(self):
        missing = [
            img["file_name"] 
            for img in self.coco.imgs.values()
            if not (self.images_dir / img["file_name"]).exists()
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images manquantes: {missing[:5]}...")
        print(f"✅ Dataset OK: {len(self.coco.imgs)} images")


# ====================
# CONFIGURATION
# ====================
class UterusSegmentationTrainer:
    """Configuration optimisée pour maximiser les vrais positifs."""
    
    def __init__(self):
        self.cfg = get_cfg()
        self._setup_cfg()

    def _setup_cfg(self):
        # Modèle de base
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        # Optimisation solver
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.BASE_LR = 2e-4
        self.cfg.SOLVER.MAX_ITER = 6000
        self.cfg.SOLVER.WARMUP_ITERS = 500
        self.cfg.SOLVER.STEPS = (4000, 5500)
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500
        
        # Accélération DataLoader
        self.cfg.DATALOADER.NUM_WORKERS = 2

        # ✅ OPTIMISATION POUR MAXIMISER LES VRAIS POSITIFS
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

        # Augmentation de données
        self.cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
        self.cfg.INPUT.MAX_SIZE_TRAIN = 1333
        self.cfg.INPUT.RANDOM_FLIP = "horizontal"

        # Output
        self.cfg.OUTPUT_DIR = "./output"
        Path(self.cfg.OUTPUT_DIR).mkdir(exist_ok=True)
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def register_datasets(self, train_json: str, train_imgs: str, val_json: str, val_imgs: str):
        """Enregistre les datasets."""
        for name in ("uterus_train", "uterus_val"):
            if name in DatasetCatalog.list():
                DatasetCatalog.remove(name)
                MetadataCatalog.remove(name)
        
        register_coco_instances("uterus_train", {}, train_json, train_imgs)
        register_coco_instances("uterus_val", {}, val_json, val_imgs)
        
        self.cfg.DATASETS.TRAIN = ("uterus_train",)
        self.cfg.DATASETS.TEST = ("uterus_val",)
        
        MetadataCatalog.get("uterus_train").set(thing_classes=["uterus"])
        MetadataCatalog.get("uterus_val").set(thing_classes=["uterus"])

    def train(self):
        """Lance l'entraînement."""
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        if is_main_process():
            trainer.checkpointer.save("model_final")
        
        return trainer


# ====================
# MAIN
# ====================
def main_training_pipeline() -> Tuple[DefaultTrainer, dict]:
    """Pipeline complet d'entraînement."""
    TRAIN_JSON = "data/train/annotations.json"
    TRAIN_IMAGES = "data/train/images"
    VAL_JSON = "data/val/annotations.json"
    VAL_IMAGES = "data/val/images"

    print("\n" + "="*60)
    print("UTERUS SEGMENTATION TRAINING")
    print("="*60)

    # Validation des datasets
    print("\n📋 Validation des datasets...")
    DatasetValidator(TRAIN_JSON, TRAIN_IMAGES).validate()
    DatasetValidator(VAL_JSON, VAL_IMAGES).validate()

    # Configuration et enregistrement
    print("\n⚙️  Configuration du modèle...")
    trainer_obj = UterusSegmentationTrainer()
    trainer_obj.register_datasets(TRAIN_JSON, TRAIN_IMAGES, VAL_JSON, VAL_IMAGES)

    # Entraînement
    print("\n🚀 Démarrage de l'entraînement...")
    print(f"   Max iterations: {trainer_obj.cfg.SOLVER.MAX_ITER}")
    print(f"   Learning rate: {trainer_obj.cfg.SOLVER.BASE_LR}")
    print(f"   Batch size: {trainer_obj.cfg.SOLVER.IMS_PER_BATCH}")
    print(f"   Device: {trainer_obj.cfg.MODEL.DEVICE}")
    
    trainer = trainer_obj.train()

    # Sauvegarde config
    with open("output/config.yaml", "w") as f:
        f.write(trainer_obj.cfg.dump())

    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*60)
    print(f"\n📁 Modèle sauvegardé: {trainer_obj.cfg.OUTPUT_DIR}/model_final.pth")
    print(f"📁 Config sauvegardée: output/config.yaml")

    return trainer, {}


if __name__ == "__main__":
    main_training_pipeline()