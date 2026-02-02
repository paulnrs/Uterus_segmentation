from pathlib import Path
import os
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from pycocotools.coco import COCO


class MaxRecallTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


class MaxRecallConfig:

    def __init__(self):
        self.cfg = get_cfg()
        self._setup_config()

    def _setup_config(self):
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
        self.cfg.SOLVER.BASE_LR = 0.001
        self.cfg.SOLVER.MAX_ITER = 1000
        self.cfg.SOLVER.WARMUP_ITERS = 100
        self.cfg.SOLVER.STEPS = (700,)
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500
        self.cfg.DATALOADER.NUM_WORKERS = 2

        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 4000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
        self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 4000
        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
        self.cfg.MODEL.RPN.NMS_THRESH = 0.8
        self.cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.7
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5

        self.cfg.INPUT.MIN_SIZE_TRAIN = (640, 704, 768, 800)
        self.cfg.INPUT.MAX_SIZE_TRAIN = 1333
        self.cfg.INPUT.RANDOM_FLIP = "horizontal"

        self.cfg.OUTPUT_DIR = "./output_max_recall"
        Path(self.cfg.OUTPUT_DIR).mkdir(exist_ok=True)

        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def register_datasets(
        self,
        train_json,
        train_images,
        val_json,
        val_images
    ):
        for name in ["uterus_train", "uterus_val"]:
            if name in DatasetCatalog.list():
                DatasetCatalog.remove(name)
                MetadataCatalog.remove(name)

        register_coco_instances("uterus_train", {}, train_json, train_images)
        register_coco_instances("uterus_val", {}, val_json, val_images)

        self.cfg.DATASETS.TRAIN = ("uterus_train",)
        self.cfg.DATASETS.TEST = ("uterus_val",)

        MetadataCatalog.get("uterus_train").set(thing_classes=["uterus"])
        MetadataCatalog.get("uterus_val").set(thing_classes=["uterus"])

        coco = COCO(train_json)
        print(f"Dataset train: {len(coco.imgs)} images, {len(coco.anns)} annotations")


def main():
    TRAIN_JSON = "data/train/annotations.json"
    TRAIN_IMAGES = "data/train/images"
    VAL_JSON = "data/val/annotations.json"
    VAL_IMAGES = "data/val/images"

    cfg_obj = MaxRecallConfig()
    cfg_obj.register_datasets(
        TRAIN_JSON,
        TRAIN_IMAGES,
        VAL_JSON,
        VAL_IMAGES
    )

    trainer = MaxRecallTrainer(cfg_obj.cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
