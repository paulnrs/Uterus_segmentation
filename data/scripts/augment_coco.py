#!/usr/bin/env python3
"""Augmente un dataset COCO en générant des images transformées et en mettant à jour les annotations."""
import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow est requis: installez-le avec `pip install pillow`.\n" + str(exc))


Point = Tuple[float, float]
Segmentation = List[List[float]]


@dataclass
class Augmentation:
    name: str
    apply_image: Callable[[Image.Image], Image.Image]
    point_transform: Optional[Callable[[float, float, int, int], Point]]


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def iter_points(flat_polygon: Sequence[float]) -> Iterable[Point]:
    it = iter(flat_polygon)
    return zip(it, it)


def format_for_suffix(suffix: str) -> Optional[str]:
    mapping = {
        '.jpg': 'JPEG',
        '.jpeg': 'JPEG',
        '.png': 'PNG',
        '.webp': 'WEBP',
    }
    return mapping.get(suffix.lower())


def reshape_segmentation(segment: Sequence[float]) -> List[float]:
    return [float(v) for v in segment]


def transform_segmentation(segmentation: Sequence[Sequence[float]], width: int, height: int,
                           point_transform: Optional[Callable[[float, float, int, int], Point]]) -> Segmentation:
    if point_transform is None:
        return [reshape_segmentation(poly) for poly in segmentation]

    transformed: Segmentation = []
    for poly in segmentation:
        coords: List[float] = []
        for x, y in iter_points(poly):
            nx, ny = point_transform(float(x), float(y), width, height)
            coords.extend([
                round(clip(nx, 0.0, float(width)), 2),
                round(clip(ny, 0.0, float(height)), 2),
            ])
        transformed.append(coords)
    return transformed


def polygon_area(coords: Sequence[float]) -> float:
    if len(coords) < 6:
        return 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) * 0.5


def segmentation_area(segmentation: Sequence[Sequence[float]]) -> float:
    return float(sum(polygon_area(poly) for poly in segmentation))


def segmentation_bbox(segmentation: Sequence[Sequence[float]]) -> List[float]:
    xs: List[float] = []
    ys: List[float] = []
    for poly in segmentation:
        xs.extend(poly[0::2])
        ys.extend(poly[1::2])
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return [round(min_x, 2), round(min_y, 2), round(max_x - min_x, 2), round(max_y - min_y, 2)]


def build_sine_lookup(amplitude: float = 0.25) -> List[int]:
    table: List[int] = []
    for i in range(256):
        normalized = i / 255.0
        shifted = normalized + amplitude * math.sin(math.pi * (normalized - 0.5))
        table.append(int(round(clip(shifted, 0.0, 1.0) * 255)))
    return table


def identity_point(x: float, y: float, _w: int, _h: int) -> Point:
    return x, y


def horizontal_flip_point(x: float, y: float, width: int, _height: int) -> Point:
    return (width - x, y)


def vertical_flip_point(x: float, y: float, _width: int, height: int) -> Point:
    return (x, height - y)


def rotate_point_builder(angle_degrees: float) -> Callable[[float, float, int, int], Point]:
    radians = math.radians(angle_degrees)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)

    def _rotate(x: float, y: float, width: int, height: int) -> Point:
        cx = width / 2.0
        cy = height / 2.0
        dx = x - cx
        dy = y - cy
        rx = dx * cos_angle - dy * sin_angle
        ry = dx * sin_angle + dy * cos_angle
        return rx + cx, ry + cy

    return _rotate


def ensure_sequence(segmentation: Sequence) -> List[Sequence[float]]:
    if not isinstance(segmentation, list):
        raise ValueError('La segmentation doit être une liste.')
    if segmentation and isinstance(segmentation[0], (int, float)):
        return [segmentation]  # type: ignore[return-value]
    return segmentation  # type: ignore[return-value]


def augment_dataset(annotation_path: Path, image_root: Path, output_annotation_path: Path,
                    output_subdir: str = 'augmented') -> None:
    coco = json.loads(annotation_path.read_text())

    images = coco.get('images', [])
    annotations = coco.get('annotations', [])

    image_lookup: dict[int, dict] = {img['id']: img for img in images}
    anns_by_image: dict[int, List[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    next_image_id = max((img['id'] for img in images), default=0) + 1
    next_ann_id = max((ann['id'] for ann in annotations), default=0) + 1

    augmented_images: List[dict] = []
    augmented_annotations: List[dict] = []

    sine_table = build_sine_lookup()

    def sine_curve(img: Image.Image) -> Image.Image:
        table = sine_table * len(img.getbands())
        return img.point(table)

    augmentations = [
        Augmentation('flipH', lambda img: img.transpose(Image.FLIP_LEFT_RIGHT), horizontal_flip_point),
        Augmentation('flipV', lambda img: img.transpose(Image.FLIP_TOP_BOTTOM), vertical_flip_point),
        Augmentation('rot15', lambda img: img.rotate(15, resample=Image.BICUBIC, expand=False), rotate_point_builder(15)),
        Augmentation('rot-15', lambda img: img.rotate(-15, resample=Image.BICUBIC, expand=False), rotate_point_builder(-15)),
        Augmentation('sineTone', sine_curve, None),
    ]

    output_root = image_root / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    for image_info in images:
        rel_path = Path(image_info['file_name'])
        candidate_paths = [image_root / rel_path]
        if not candidate_paths[0].exists():
            candidate_paths.append(image_root / rel_path.name)
        image_path = next((p for p in candidate_paths if p.exists()), None)
        if image_path is None:
            print(f"[AVERTISSEMENT] Image introuvable pour {image_info['file_name']}, augmentation ignorée.")
            continue

        with Image.open(image_path) as img:
            base_image = img.convert('RGB')
            suffix = image_path.suffix or '.jpg'
            parent_fragment = rel_path.parent if str(rel_path.parent) not in ('', '.') else Path()

            for aug in augmentations:
                new_image = aug.apply_image(base_image)
                new_stem = f"{rel_path.stem}__{aug.name}"
                new_name = new_stem + suffix
                destination_rel = (parent_fragment / output_subdir / new_name) if parent_fragment else Path(output_subdir) / new_name
                destination_path = image_root / destination_rel
                destination_path.parent.mkdir(parents=True, exist_ok=True)

                pil_format = format_for_suffix(suffix) or base_image.format
                save_kwargs = {'format': pil_format} if pil_format else {}
                new_image.save(destination_path, **save_kwargs)

                new_image_id = next_image_id
                next_image_id += 1

                new_image_info = {
                    'id': new_image_id,
                    'width': new_image.width,
                    'height': new_image.height,
                    'file_name': str(destination_rel.as_posix()),
                    'license': image_info.get('license', 0),
                    'flickr_url': image_info.get('flickr_url', ''),
                    'coco_url': image_info.get('coco_url', ''),
                    'date_captured': image_info.get('date_captured', 0),
                }
                augmented_images.append(new_image_info)

                for original_ann in anns_by_image.get(image_info['id'], []):
                    if original_ann.get('iscrowd', 0) == 1:
                        print(f"[INFO] Annotation {original_ann['id']} ignorée (iscrowd=1).")
                        continue

                    original_seg = ensure_sequence(original_ann.get('segmentation', []))
                    transformed_seg = transform_segmentation(
                        original_seg,
                        new_image.width,
                        new_image.height,
                        aug.point_transform or identity_point,
                    )
                    new_bbox = original_ann.get('bbox', [0.0, 0.0, 0.0, 0.0])
                    new_area = original_ann.get('area', 0.0)

                    if aug.point_transform is not None:
                        new_bbox = segmentation_bbox(transformed_seg)
                        new_area = segmentation_area(transformed_seg)

                    new_ann = {
                        'id': next_ann_id,
                        'image_id': new_image_id,
                        'category_id': original_ann['category_id'],
                        'segmentation': transformed_seg,
                        'area': new_area,
                        'bbox': new_bbox,
                        'iscrowd': original_ann.get('iscrowd', 0),
                    }
                    if 'attributes' in original_ann:
                        new_ann['attributes'] = deepcopy(original_ann['attributes'])
                    for key in ('num_keypoints', 'keypoints'):
                        if key in original_ann:
                            new_ann[key] = deepcopy(original_ann[key])

                    augmented_annotations.append(new_ann)
                    next_ann_id += 1

    coco['images'].extend(augmented_images)
    coco['annotations'].extend(augmented_annotations)
    if coco.get('info'):
        info_description = coco['info'].get('description') or ''
        note = ' (augmenté)' if '(augmenté)' not in info_description else ''
        coco['info']['description'] = info_description + note

    output_annotation_path.write_text(json.dumps(coco, ensure_ascii=False))

    print(f"Images originales: {len(images)}")
    print(f"Nouvelles images: {len(augmented_images)}")
    print(f"Annotations originales: {len(annotations)}")
    print(f"Nouvelles annotations: {len(augmented_annotations)}")
    print(f"Annotation augmentée sauvegardée dans: {output_annotation_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Augmente un dataset COCO (images + annotations).')
    parser.add_argument('--annotation', type=Path, default=Path('annotations/instances_Train.json'))
    parser.add_argument('--images', type=Path, default=Path('images'))
    parser.add_argument('--output-annotation', type=Path, default=Path('annotations/instances_Train_augmented.json'))
    parser.add_argument('--output-subdir', type=str, default='augmented')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augment_dataset(args.annotation, args.images, args.output_annotation, args.output_subdir)


if __name__ == '__main__':
    main()