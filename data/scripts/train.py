from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import cv2

from detectron2.engine import hooks
from detectron2.utils.events import get_event_storage
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.comm import is_main_process

from pycocotools.coco import COCO
from pycocotools import mask as mask_util

class EarlyStoppingHook(hooks.HookBase):
    """Early stopping basé sur une métrique du storage"""
    
    def __init__(self, patience: int = 5, metric_name: str = "dice", mode: str = "max"):
        """
        Args:
            patience: Nombre d'évaluations sans amélioration avant arrêt
            metric_name: Nom de la métrique dans le storage (ex: "dice")
            mode: "max" pour maximiser, "min" pour minimiser
        """
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = None
        self.counter = 0
        
        # Fonction de comparaison
        if mode == "max":
            self.is_better = lambda new, best: new > best
        else:
            self.is_better = lambda new, best: new < best

    def after_step(self):
        # Récupérer le storage
        storage = get_event_storage()
        
        # ✅ Vérifier si la métrique existe dans le storage
        if self.metric_name not in storage._latest_scalars:
            return
        
        # Lire la métrique actuelle
        current_metric = storage._latest_scalars[self.metric_name][0]
        
        # Premier enregistrement
        if self.best_metric is None:
            self.best_metric = current_metric
            self.counter = 0
            print(f"📊 Early stopping initialisé: {self.metric_name} = {current_metric:.4f}")
            return
        
        # Vérifier amélioration
        if self.is_better(current_metric, self.best_metric):
            improvement = abs(current_metric - self.best_metric)
            self.best_metric = current_metric
            self.counter = 0
            print(f"✅ Amélioration: {self.metric_name} = {current_metric:.4f} (+{improvement:.4f})")
        else:
            self.counter += 1
            print(f"⏸️  Pas d'amélioration ({self.counter}/{self.patience}): "
                  f"{self.metric_name} = {current_metric:.4f} (best: {self.best_metric:.4f})")
        
        # Déclencher l'arrêt
        if self.counter >= self.patience:
            print(f"\n🛑 EARLY STOPPING DÉCLENCHÉ!")
            print(f"   Pas d'amélioration depuis {self.patience} évaluations")
            print(f"   Meilleur {self.metric_name}: {self.best_metric:.4f}")
            print(f"   Arrêt à l'itération {self.trainer.iter}\n")
            
            # ✅ Arrêter l'entraînement
            raise StopIteration("Early stopping triggered")


class DiceEvaluator:
    """Évaluateur de Dice simplifié qui travaille directement avec les données"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dices = []
        self.total_preds = 0
        self.total_gts = 0

    def process_one(self, data_dict, prediction):
        """Process une seule image"""
        h, w = data_dict["height"], data_dict["width"]

        # Construction du masque de prédiction
        if len(prediction["instances"]) == 0:
            pred_mask = np.zeros((h, w), dtype=bool)
        else:
            pred_masks = prediction["instances"].pred_masks.cpu().numpy()
            pred_mask = pred_masks.any(axis=0)

        # Construction du masque GT
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
        
        # Compteurs
        self.total_preds += pred_mask.sum()
        self.total_gts += gt_mask.sum()
        
        # Calcul du Dice
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
            print(f"Total PRED pixels: {self.total_preds:,}")
            print(f"Total GT pixels  : {self.total_gts:,}")
            print(f"{'='*60}\n")

        return {"dice": mean_dice}


def dice_eval_fn(cfg, model):
    """
    Fonction d'évaluation qui charge les données avec annotations
    et fait l'inférence manuellement (comme le script offline)
    """
    evaluator = DiceEvaluator()
    
    # Charger les données brutes avec annotations
    dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    
    model.eval()
    with torch.no_grad():
        for data in dataset_dicts:
            # Charger l'image
            img = cv2.imread(data["file_name"])
            if img is None:
                print(f"⚠️ Cannot read image: {data['file_name']}")
                continue
                
            height, width = img.shape[:2]
            
            # Préparer l'input pour le modèle
            image_tensor = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            inputs = [{
                "image": image_tensor,
                "height": height,
                "width": width
            }]
            
            # Prédire
            outputs = model(inputs)
            
            # Évaluer avec les annotations originales
            evaluator.process_one(data, outputs[0])
    
    return evaluator.evaluate()


class DiceTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks_list = super().build_hooks()

        eval_hook = hooks.EvalHook(
            eval_period=250,
            eval_function=lambda: dice_eval_fn(self.cfg, self.model),
        )

        best_checkpointer = hooks.BestCheckpointer(
            eval_period=250,
            checkpointer=self.checkpointer,
            val_metric="dice",
            mode="max",
            file_prefix="model_best",
        )
        
        early_stop = EarlyStoppingHook(
            patience=10,
            metric_name="dice",
            mode="max"
        )

        hooks_list.insert(-1, eval_hook)
        hooks_list.insert(-1, best_checkpointer)
        hooks_list.insert(-1, early_stop)  

        return hooks_list


class DatasetValidator:
    def __init__(self, annotations_path, images_dir):
        self.coco = COCO(annotations_path)
        self.images_dir = Path(images_dir)

    def validate(self):
        missing = [img["file_name"] for img in self.coco.imgs.values()
                   if not (self.images_dir / img["file_name"]).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images manquantes: {missing[:5]}...")
        print(f"Dataset OK: {len(self.coco.imgs)} images")


class UterusSegmentationTrainer:
    def __init__(self):
        self.cfg = get_cfg()
        self._setup_cfg()

    def _setup_cfg(self):
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.BASE_LR = 2e-4
        self.cfg.SOLVER.MAX_ITER = 6000
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500
        self.cfg.SOLVER.WARMUP_ITERS = 500
        self.cfg.SOLVER.STEPS = (4000, 5500)
        self.cfg.SOLVER.GAMMA = 0.1

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

        self.cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
        self.cfg.INPUT.MAX_SIZE_TRAIN = 1333
        self.cfg.INPUT.RANDOM_FLIP = "horizontal"

        self.cfg.OUTPUT_DIR = "./output"
        Path(self.cfg.OUTPUT_DIR).mkdir(exist_ok=True)

        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def register_datasets(self, train_json, train_imgs, val_json, val_imgs):
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
        trainer = DiceTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        if is_main_process():
            trainer.checkpointer.save("model_final")
        return trainer


def main_training_pipeline() -> Tuple[DefaultTrainer, dict]:
    TRAIN_JSON = "data/train/annotations.json"
    TRAIN_IMAGES = "data/train/images"
    VAL_JSON = "data/val/annotations.json"
    VAL_IMAGES = "data/val/images"

    DatasetValidator(TRAIN_JSON, TRAIN_IMAGES).validate()
    DatasetValidator(VAL_JSON, VAL_IMAGES).validate()

    trainer_obj = UterusSegmentationTrainer()
    trainer_obj.register_datasets(TRAIN_JSON, TRAIN_IMAGES, VAL_JSON, VAL_IMAGES)
    trainer = trainer_obj.train()

    with open("output/config.yaml", "w") as f:
        f.write(trainer_obj.cfg.dump())

    print("Training complete. Models saved in: output/")
    print("   • model_best.pth")
    print("   • model_final.pth")

    return trainer, {}


if __name__ == "__main__":
    main_training_pipeline()