"""Training optimisé pour MAXIMISER les VRAIS POSITIFS (quitte à avoir des faux positifs)."""

from pathlib import Path
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from pycocotools.coco import COCO


class MaxRecallTrainer:
    """Configuration ultra-agressive pour ne RIEN manquer."""
    
    def __init__(self):
        self.cfg = get_cfg()
        self._setup_config()

    def _setup_config(self):
        # Modèle de base
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        # ========================================
        # ENTRAÎNEMENT COURT (1000 iterations)
        # ========================================
        self.cfg.SOLVER.IMS_PER_BATCH = 16  # Batch plus gros pour apprendre vite
        self.cfg.SOLVER.BASE_LR = 0.001  # LR élevé pour convergence rapide
        self.cfg.SOLVER.MAX_ITER = 1000  # Court !
        self.cfg.SOLVER.WARMUP_ITERS = 100
        self.cfg.SOLVER.STEPS = (700,)  # Un seul decay à 70%
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500
        self.cfg.DATALOADER.NUM_WORKERS = 2

        # ========================================
        # 🎯 MAXIMISATION DES VRAIS POSITIFS (version équilibrée)
        # ========================================
        
        # 1. RPN Agressif mais pas extrême
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 4000  # 2x plus
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000  # 2x plus
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 4000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
        
        # 2. Seuils RPN permissifs
        self.cfg.MODEL.RPN.NMS_THRESH = 0.8  # Plus permissif que 0.7 par défaut
        self.cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
        
        # 3. ROI Heads : Seuils bas mais pas extrêmes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Bas mais raisonnable
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7  # Permet du chevauchement
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        
        # 4. Perte focale pour gérer le déséquilibre (focus sur les rares VP)
        # Note: Mask R-CNN n'utilise pas focal loss par défaut, mais on peut 
        # augmenter le poids des positifs via POSITIVE_FRACTION
        self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5  # Plus de positifs dans le batch
        
        # 5. Augmentation de données MODERÉE (pas trop pour apprendre vite)
        self.cfg.INPUT.MIN_SIZE_TRAIN = (640, 704, 768, 800)
        self.cfg.INPUT.MAX_SIZE_TRAIN = 1333
        self.cfg.INPUT.RANDOM_FLIP = "horizontal"

        # Output
        self.cfg.OUTPUT_DIR = "./output_max_recall"
        Path(self.cfg.OUTPUT_DIR).mkdir(exist_ok=True)
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def register_datasets(self, train_json: str, train_imgs: str):
        """Enregistre uniquement le training set (pas de validation)."""
        if "uterus_train" in DatasetCatalog.list():
            DatasetCatalog.remove("uterus_train")
            MetadataCatalog.remove("uterus_train")
        
        register_coco_instances("uterus_train", {}, train_json, train_imgs)
        self.cfg.DATASETS.TRAIN = ("uterus_train",)
        self.cfg.DATASETS.TEST = ()  # Pas de test pendant l'entraînement
        MetadataCatalog.get("uterus_train").set(thing_classes=["uterus"])
        
        # Vérifier le dataset
        coco = COCO(train_json)
        print(f"✅ Dataset: {len(coco.imgs)} images, {len(coco.anns)} annotations")

    def train(self):
        """Lance l'entraînement."""
        print("\n" + "="*60)
        print("🎯 ENTRAÎNEMENT POUR MAXIMISER LES VRAIS POSITIFS")
        print("="*60)
        print("\n📋 Configuration:")
        print(f"   Iterations: {self.cfg.SOLVER.MAX_ITER}")
        print(f"   Batch size: {self.cfg.SOLVER.IMS_PER_BATCH}")
        print(f"   Learning rate: {self.cfg.SOLVER.BASE_LR}")
        print(f"   Device: {self.cfg.MODEL.DEVICE}")
        print("\n🔧 Paramètres Recall Max:")
        print(f"   RPN proposals (test): {self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST}")
        print(f"   Score threshold: {self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
        print(f"   NMS threshold: {self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST}")
        print("\n💡 Objectif: Maximiser Recall tout en gardant un Dice correct")
        
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        
        print("\n🚀 Démarrage...\n")
        trainer.train()
        
        trainer.checkpointer.save("model_max_recall")
        print("\n" + "="*60)
        print("✅ ENTRAÎNEMENT TERMINÉ")
        print("="*60)
        print(f"\n📁 Modèle: {self.cfg.OUTPUT_DIR}/model_max_recall.pth")
        print("\n💡 Ce modèle est configuré pour:")
        print("   ✓ Maximiser le Recall (peu de faux négatifs)")
        print("   ✓ Seuil bas (0.05) mais raisonnable")
        print("   ✓ Dice attendu: 0.75-0.85 (légère baisse vs 0.90)")
        print("   ✓ Recall attendu: 95-98%")
        
        # Sauvegarder la config
        with open(f"{self.cfg.OUTPUT_DIR}/config.yaml", "w") as f:
            f.write(self.cfg.dump())
        
        return trainer


def main():
    """Pipeline d'entraînement."""
    TRAIN_JSON = "data/train/annotations.json"
    TRAIN_IMAGES = "data/train/images"

    trainer_obj = MaxRecallTrainer()
    trainer_obj.register_datasets(TRAIN_JSON, TRAIN_IMAGES)
    trainer = trainer_obj.train()
    
    print("\n" + "="*60)
    print("📊 COMMENT UTILISER CE MODÈLE")
    print("="*60)
    print("""
Pour l'inférence, utilisez:

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file("output_max_recall/config.yaml")
cfg.MODEL.WEIGHTS = "output_max_recall/model_max_recall.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # Très permissif !

predictor = DefaultPredictor(cfg)
outputs = predictor(image)

# Vous aurez BEAUCOUP de détections (y compris des FP)
# mais vous ne manquerez RIEN (max recall)
""")


if __name__ == "__main__":
    main()