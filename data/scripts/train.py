"""Training pipeline for Detectron2 uterus segmentation - Version Optimisée."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, hooks
from detectron2.utils.comm import is_main_process
from detectron2.utils.events import get_event_storage

from pycocotools.coco import COCO
from pycocotools import mask as mask_util


# ====================
# EARLY STOPPING
# ====================
class EarlyStoppingHook(hooks.HookBase):
    """Early stopping basé sur le Dice score de validation."""
    
    def __init__(self, patience: int = 10, metric_name: str = "dice"):
        self.patience = patience
        self.metric_name = metric_name
        self.best_metric = None
        self.counter = 0

    def after_step(self):
        storage = get_event_storage()
        
        if self.metric_name not in storage._latest_scalars:
            return
        
        current_metric = storage._latest_scalars[self.metric_name][0]
        
        if self.best_metric is None:
            self.best_metric = current_metric
            self.counter = 0
            print(f"📊 Early stopping initialisé: {self.metric_name} = {current_metric:.4f}")
            return
        
        if current_metric > self.best_metric:
            improvement = current_metric - self.best_metric
            self.best_metric = current_metric
            self.counter = 0
            print(f"✅ Amélioration: {self.metric_name} = {current_metric:.4f} (+{improvement:.4f})")
        else:
            self.counter += 1
            print(f"⏸️  Pas d'amélioration ({self.counter}/{self.patience}): "
                  f"{self.metric_name} = {current_metric:.4f} (best: {self.best_metric:.4f})")
        
        if self.counter >= self.patience:
            print(f"\n🛑 EARLY STOPPING à iter {self.trainer.iter}")
            print(f"   Meilleur {self.metric_name}: {self.best_metric:.4f}\n")
            raise StopIteration("Early stopping triggered")


# ====================
# DICE EVALUATOR
# ====================
class SimpleDiceEvaluator:
    """Évaluateur Dice - même logique que votre script offline."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dices = []

    def process_one(self, data_dict, prediction):
        """Traite une image - identique à votre code offline."""
        h, w = data_dict["height"], data_dict["width"]

        # Masque de prédiction
        if len(prediction["instances"]) == 0:
            pred_mask = np.zeros((h, w), dtype=bool)
        else:
            pred_masks = prediction["instances"].pred_masks.cpu().numpy()
            pred_mask = pred_masks.any(axis=0)

        # Masque GT - EXACTEMENT comme votre script offline
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        for ann in data_dict.get("annotations", []):
            seg = ann.get("segmentation", [])
            
            if isinstance(seg, dict):  # RLE
                m = mask_util.decode(seg)
                gt_mask = np.maximum(gt_mask, m.astype(np.uint8))
            
            elif isinstance(seg, list):  # Polygones
                for poly in seg:
                    poly_np = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(gt_mask, [poly_np], 1)

        gt_mask = gt_mask.astype(bool)
        
        # Dice
        inter = np.logical_and(pred_mask, gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice = (2 * inter / union) if union > 0 else 0.0
        self.dices.append(dice)

    def evaluate(self):
        if not self.dices:
            return {"dice": 0.0}

        mean_dice = float(np.mean(self.dices))

        if is_main_process():
            print(f"\n{'='*60}")
            print(f"DICE EVALUATION")
            print(f"{'='*60}")
            print(f"Mean Dice: {mean_dice:.4f}")
            print(f"Min Dice : {np.min(self.dices):.4f}")
            print(f"Max Dice : {np.max(self.dices):.4f}")
            print(f"Std Dice : {np.std(self.dices):.4f}")
            print(f"N images : {len(self.dices)}")
            print(f"{'='*60}\n")

        return {"dice": mean_dice}


def dice_eval_fn(cfg, model):
    """Évaluation Dice - charge les données comme votre script offline."""
    evaluator = SimpleDiceEvaluator()
    dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    
    model.eval()
    with torch.no_grad():
        for data in dataset_dicts:
            img = cv2.imread(data["file_name"])
            if img is None:
                continue
                
            height, width = img.shape[:2]
            image_tensor = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            outputs = model(inputs)
            evaluator.process_one(data, outputs[0])
    
    model.train()
    return evaluator.evaluate()


# ====================
# TRAINER
# ====================
class DiceTrainer(DefaultTrainer):
    """Trainer avec évaluation Dice et early stopping."""
    
    def build_hooks(self):
        hooks_list = super().build_hooks()

        # Évaluation Dice toutes les 250 itérations
        eval_hook = hooks.EvalHook(
            eval_period=250,
            eval_function=lambda: dice_eval_fn(self.cfg, self.model),
        )

        # Sauvegarde du meilleur modèle
        best_checkpointer = hooks.BestCheckpointer(
            eval_period=250,
            checkpointer=self.checkpointer,
            val_metric="dice",
            mode="max",
            file_prefix="model_best",
        )
        
        # Early stopping
        early_stop = EarlyStoppingHook(patience=10, metric_name="dice")

        hooks_list.insert(-1, eval_hook)
        hooks_list.insert(-1, best_checkpointer)
        hooks_list.insert(-1, early_stop)

        return hooks_list


# ====================
# VALIDATOR
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
    """Configuration et entraînement optimisés pour maximiser les vrais positifs."""
    
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

        # ✅ OPTIMISATION POUR MAXIMISER LES VRAIS POSITIFS
        # Augmenter le nombre de propositions pour ne rien manquer
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000  # ↑ Plus de propositions
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000  # ↑
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 3000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
        
        # Seuils très permissifs pour capturer toutes les détections
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # ↓ Seuil très bas
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7  # ↑ Permet plus de chevauchement
        
        # Plus de ROIs par image pour ne rien manquer
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

        # Augmentation de données pour robustesse
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
        trainer = DiceTrainer(self.cfg)
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
    print(f"\n📁 Modèles sauvegardés dans: {trainer_obj.cfg.OUTPUT_DIR}/")
    print("   • model_best.pth   (meilleur Dice)")
    print("   • model_final.pth  (dernier checkpoint)")
    print("   • config.yaml      (configuration)")

    return trainer, {}


if __name__ == "__main__":
    main_training_pipeline()