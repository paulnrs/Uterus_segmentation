# Uterus_segmentation
# Projet Uterus Segmentation – Rapport Technique

Ce document décrit l'ensemble du pipeline construit pour la segmentation de l'utérus à partir d'échographies. Il couvre la préparation de l'environnement, l'organisation des données, la validation des annotations, l'entraînement, l'évaluation, ainsi que les points clés pour la reproduction et le suivi des résultats.

## 1. Installation & Préparation de l'Environnement

### 1.1 Prérequis logiciels
- Python 3.11 ou 3.12 (Detectron2 ne supporte pas encore Python 3.13).
- Pip, Git, et un compilateur C/C++ (nécessaires pour construire Detectron2 si les roues binaires ne sont pas disponibles).
- (Optionnel) GPU CUDA / cuDNN pour accélérer l'entraînement ; sur CPU, réduire le batch size.

### 1.2 Création d'un environnement virtuel
```bash
python3.11 -m venv venv_detectron2
source venv_detectron2/bin/activate  # macOS/Linux
# ou sous Windows
venv_detectron2\Scripts\activate
```

### 1.3 Installation de PyTorch
Pour CPU ou Apple Silicon :
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Adapter l'URL si vous disposez d'une GPU NVIDIA (ex.: `.../cu118`).

### 1.4 Installation de Detectron2
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 1.5 Dépendances complémentaires
```bash
pip install opencv-python pillow matplotlib pycocotools tensorboard pandas scikit-image
```

### 1.6 Configuration Matplotlib (macOS)
Matplotlib peut échouer si `~/.matplotlib` n'est pas accessible ou si aucun backend graphique n'est disponible.
```bash
mkdir -p /FEMNOV/mpl_cache
export MPLCONFIGDIR=/FEMNOV/mpl_cache
export MPLBACKEND=Agg
```
Ces variables peuvent être définies dans le shell avant tout lancement du script d'entraînement.

## 2. Organisation du Projet

```
FEMNOV/
├── data/
│   ├── train/
│   │   ├── images/               # 1605 images (80 %)
│   │   └── annotations.json      # COCO, 1605 annotations
│   ├── val/
│   │   ├── images/               # 200 images (10 %)
│   │   └── annotations.json      # COCO, 200 annotations
│   ├── test/
│   │   ├── images/               # 200 images (10 %)
│   │   └── annotations.json      # COCO, 200 annotations
│   └── ...                       # fichiers éventuels (annotations_augmented.json, etc.)
├── data/scripts/
│   ├── train.py                  # pipeline complet (validation + train + évaluation)
│   ├── evaluate.py               # helpers d’évaluation & inférence
│   └── ...
├── output/
│   ├── model_final.pth
│   ├── config.yaml
│   ├── metrics.json
│   ├── coco_instances_results.json
│   ├── instances_predictions.pth
│   └── events.out.tfevents.*     # logs TensorBoard
└── docs/
    └── PROJET_UTERUS.md          # rapport courant
```

Les images proviennent de `13k_images/images/augmentations`. Répartition effectuée avec une graine fixe (42) via un script Python qui copie 1605 fichiers en train, 200 en validation et 200 en test.

Les annotations COCO ont été reconstruites par split à partir de `annotations/instances_Train_augmented.json` du dataset source avec remappage des identifiants et uniformisation de la catégorie (`"uterus"`).

## 3. Nettoyage et Reconstitution des Annotations

Étapes réalisées pour aligner les annotations avec les nouvelles répartitions :
1. Suppression préalable des images dans `data/train|val|test/images`.
2. Copie 80/10/10 depuis `13k_images/images/augmentations`.
3. Génération de nouveaux `annotations.json` en ne conservant que les entrées correspondant aux images présentes. Les champs `id`, `image_id`, `file_name` et la catégorie `"uterus"` sont régénérés/normalisés.
4. Ajustement final des noms de catégorie (`"uterus"` en minuscule) pour respecter les contraintes Detectron2.

Statistiques sur les annotations d'entraînement :
- Aire min : ~3961 px² ; aire max : ~24 154 px² ; aire moyenne : ~12 395 px².
- Points par polygone : min 37, max 324, moyenne ~111.

## 4. Validation du Dataset (DatasetValidator)

Module situé dans `data/scripts/train.py` (classe `DatasetValidator`). Fonctionnalités :
- Décompte des images/annotations, affichage des catégories.
- Vérification de la présence physique des fichiers images.
- Analyse des aires et du nombre de points par polygone.
- Détection de problèmes courants : polygones dégénérés, bbox invalides, aire ≤ 0.
- `visualize_samples(num_samples=4)` : sélection aléatoire d'images avec superposition des masques et bounding boxes (nécessite Matplotlib).

Utilisation type :
```python
validator = DatasetValidator('data/train/annotations.json', 'data/train/images')
validator.validate_dataset()
validator.visualize_samples()
```

## 5. Pipeline d'Entraînement (train.py)

### 5.1 Configuration Detectron2
- **Fichier de base** `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` : charge l’architecture Mask R-CNN backbone ResNet-50 + FPN et les hyperparamètres COCO d’origine.
- **Poids initiaux** `MODEL.WEIGHTS` : checkpoint pré‑entraîné sur COCO pour accélérer la convergence.
- **Nombre de classes** `MODEL.ROI_HEADS.NUM_CLASSES = 1` : nous n’avons qu’une classe (« uterus »). À modifier si vous ajoutez d’autres structures anatomiques.
- **Batch global** `SOLVER.IMS_PER_BATCH = 16` : nombre d’images traitées simultanément (sur CPU, réduire si la RAM est limitée ; sur GPU, adapter à la VRAM disponible).
- **Taux d’apprentissage** `SOLVER.BASE_LR = 2.5e-4` : pas d’optimisation initial. Pour un batch différent, scaler linéairement (ex. batch 8 → LR ≈ 1.25e-4).
- **Planning** `SOLVER.MAX_ITER = 3000`, `SOLVER.STEPS = [2000, 2500]`, `SOLVER.GAMMA = 0.1` : nombre d’itérations max et décroissance du LR (LR × 0.1) aux itérations indiquées. Ajustez si votre dataset est plus petit/grand.
- **Augmentations test-time** `TEST.AUG` : active des redimensionnements/flip en validation pour un léger gain de robustesse. Désactivez en cas de contrainte de temps.
- **Dossier sortie** `OUTPUT_DIR = ./output` : emplacement des checkpoints, logs et métriques. Modifiez si vous souhaitez séparer plusieurs expériences.
- **Période d’évaluation** `TEST.EVAL_PERIOD = 200` : fréquence (en itérations) à laquelle le modèle est évalué et la métrique `segm/AP` calculée (nécessaire à l’early stopping).
- **Device** `MODEL.DEVICE = "cpu"` : impose le calcul sur CPU pour compatibilité macOS. Basculer sur `"cuda"` si un GPU NVIDIA est disponible.

### 5.2 Early Stopping
- **`metric`** : `segm/AP` → métrique Monitorée. On peut utiliser `bbox/AP` ou toute valeur loggée par Detectron2 (`storage.put_scalar`).
- **`patience = 5`** : nombre d’évaluations consécutives sans amélioration avant arrêt. Augmentez pour laisser plus de temps au modèle.
- **`min_delta = 1e-4`** : amélioration minimale considérée comme significative. Ajustez en fonction de la variabilité de la métrique.
- **`warmup_iters = TEST.EVAL_PERIOD`** : évite un arrêt prématuré avant la première évaluation complète.
- **`maximize = True`** : indique que la métrique doit augmenter (utile si vous surveillez une perte où `maximize=False`).
- Le `EarlyStoppingTrainer` injecte ce hook après ceux de Detectron2 pour interrompre l’entraînement (`trainer.iter = trainer.max_iter`) tout en conservant les checkpoints existants.

### 5.3 Enregistrement des datasets
`register_datasets` :
- Nettoie les entrées existantes dans `DatasetCatalog` et `MetadataCatalog`.
- Enregistre train/val via `register_coco_instances` avec les JSON rebuild.
- Définit `thing_classes = ['uterus']` pour assurer cohérence.

### 5.4 Fonction `main_training_pipeline`
1. Validation du dataset train (statistiques + visualization).
2. Configuration du modèle via `UterusSegmentationTrainer`.
3. Enregistrement des datasets train/val, affichage de confirmation.
4. Entraînement : affiche progression (`MAX_ITER`, LR, batch size, dossier output). Early stopping peut interrompre avant 3000 itérations.
5. Évaluation : instancie `ModelEvaluator`, exécute `evaluate_on_validation()` (rapporte AP segmentation/bbox, AP50, AP75).
6. Sauvegarde la configuration finale (`output/config.yaml`).
7. Message de fin + création d’un test d’inférence si `data/test/images/sample.jpg` existe.

Exécution :
```bash
MPLCONFIGDIR=/FEMNOV/mpl_cache MPLBACKEND=Agg \
python data/scripts/train.py
```

### 5.5 Suivi des métriques (metrics.json & TensorBoard)
- `output/metrics.json` : chaque ligne JSON capture les pertes et précisions par itération.
  - Sur le run en cours : 25 entrées, `total_loss` [1.09 → 1.59], `mask_rcnn/accuracy` ~0.79 à la dernière itération (99), `fast_rcnn/cls_accuracy` ~0.996.
- `output/events.out.tfevents.*` : logs TensorBoard ; lancer `tensorboard --logdir output` pour visualiser les courbes.

## 6. Évaluation & Inférence (evaluate.py)

### 6.1 `ModelEvaluator`
- Clone la configuration en entrée, reconstruit un `DefaultPredictor` et lance `COCOEvaluator` sur le dataset de validation.`
- Affiche AP segmentation et bbox (moyenne, AP50, AP75).
- Peut être réutilisé après entraînement pour recharger `model_final.pth`.

### 6.2 `UterusSegmentationInference`
- Charge `model_final.pth` et une configuration (par défaut celle du model zoo ajustée à 1 classe).
- `segment_image(path)` : renvoie masques, scores, nombre de détections.
- `process_batch(image_paths, save_results=True, output_dir=...)` : itère sur un lot et sauvegarde les principaux masques sous forme d'images binaires.

## 7. Résultats Actuels

- Répartition : 1605 images train / 200 val / 200 test (tous COCO alignés).
- Dernier run (99 itérations) :
  - `total_loss` = 1.09, `mask_rcnn/accuracy` ≈ 0.788, `fast_rcnn/cls_accuracy` ≈ 0.996.
  - Historiquement : `total_loss` min 1.09, max 1.59 ; `mask accuracy` en hausse sur les dernières itérations.
- `output/coco_instances_results.json` contient 300 prédictions (extraits de l'évaluation).
- Artifacts disponibles : `model_final.pth`, `instances_predictions.pth`, `config.yaml`, `metrics.json`, logs TensorBoard.

## 8. Conseils & Prochaines Étapes

1. **Validation complète** : relancer `python data/scripts/train.py` pour confirmer la convergence suite aux modifications (batch 16, early stopping, dataset étendu).
2. **Analyse TensorBoard** : lancer `tensorboard --logdir output` pour visualiser les courbes de perte, lr, et les éventuelles métriques d'arrêt.
3. **Evaluation additionnelle** : écrire un script/in notebook pour évaluer sur le split test (non couvert par défaut dans `train.py`).
4. **Optimisation** :
   - Ajuster `IMS_PER_BATCH`, `BASE_LR`, `MAX_ITER` selon les ressources.
   - Désactiver `TEST.AUG` si la latence est critique.
   - Modifier les paramètres de l'early stopping (patience, métrique) suivant vos attentes.
5. **Préproduction** : utiliser `UterusSegmentationInference` pour créer des workflows d'inférence batch ou temps réel (ex.: sauvegarde des masques, scoring des patients).
6. **Gestion des données** : documenter la provenance et la transformation des images (origine `13k_images`, augmentation déjà réalisée). Garantir la traçabilité si d'autres splits sont créés.
7. **Documentation continue** : compléter ce rapport avec les métriques finales obtenues après un entraînement full (ex.: AP segmentation sur val/test, Dice moyen, etc.).

## 9. Références
- [Detectron2](https://github.com/facebookresearch/detectron2) – Framework développé par Facebook AI Research.
- [Format COCO](https://cocodataset.org/#format-data) – Spécification des annotations.
- [PyTorch](https://pytorch.org/) – Backend pour Detectron2.

---
