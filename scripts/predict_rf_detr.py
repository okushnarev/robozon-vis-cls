import argparse
import json
from pathlib import Path

import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from supervision.metrics import MeanAveragePrecision
from tqdm import tqdm

# Map strings to the appropriate class definitions
MODEL_MAP = {
    'RFDETRNano':   RFDETRNano,
    'RFDETRSmall':  RFDETRSmall,
    'RFDETRMedium': RFDETRMedium,
    'RFDETRBase':   RFDETRBase,
    'RFDETRLarge':  RFDETRLarge,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with RF-DETR on a Custom COCO Dataset')
    parser.add_argument(
        '--ckpt-dir', '-c',
        type=Path,
        required=True,
        help='Path to the folder with all checkpoints and configs'

    )
    parser.add_argument(
        '--ds-dir', '-d',
        type=Path,
        default=None,
        help='Path to dataset to perform predict on. If None then \'test\' dataset from \'train_config\' is used'
    )

    parser.add_argument(
        '--out-dir', '-o',
        type=Path,
        default=Path('output/default'),
        help='Path to store detection results, images, metrics',
    )

    parser.add_argument(
        '--ckpt-type',
        type=str,
        default='checkpoint_best_total',
        help="Checkpoint type to load "
             "['checkpoint_best_ema', 'checkpoint_best_total', 'checkpoint_best_regular', 'last_ema', 'last', etc]",
    )
    return parser.parse_args()


def predict():
    args = parse_args()
    with open(args.ckpt_dir / 'training_config.json', 'r') as f:
        config = json.load(f)

    ModelClass = MODEL_MAP[config['model_config']['model_name']]
    model = ModelClass(pretrain_weights=args.ckpt_dir / f'{args.ckpt_type}.pth')
    model.optimize_for_inference()

    if args.ds_dir:
        ds_path = args.ds_dir
    else:
        ds_path = Path(config['train_config']['dataset_dir']) / 'test'

    img_path = args.out_dir / 'images'
    true_img_path = img_path / 'true'
    pred_img_path = img_path / 'pred'

    true_img_path.mkdir(exist_ok=True, parents=True)
    pred_img_path.mkdir(exist_ok=True, parents=True)

    ds = sv.DetectionDataset.from_coco(
        images_directory_path=ds_path,
        annotations_path=f'{ds_path}/_annotations.coco.json',
    )

    path, image, annotations = ds[0]
    image = Image.open(path)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale)

    targets = []
    predictions = []
    for path, _, annotations in tqdm(ds):
        image = Image.open(path)
        detections = model.predict(image, threshold=0.5)

        if text_scale is None:
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
        if thickness is None:
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

        annotations_labels = [
            f'{ds.classes[class_id]}'
            for class_id
            in annotations.class_id
        ]

        ann_image = image.copy()
        ann_image = bbox_annotator.annotate(ann_image, annotations)
        ann_image = label_annotator.annotate(ann_image, annotations, annotations_labels)
        ann_image.save(
            true_img_path / f'{Path(path).stem}.png'
        )

        detections_labels = [
            f'{ds.classes[class_id]} {confidence:.2f}'
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        det_image = image.copy()
        det_image = bbox_annotator.annotate(det_image, detections)
        det_image = label_annotator.annotate(det_image, detections, detections_labels)
        det_image.save(
            pred_img_path / f'{Path(path).stem}.png'
        )

        targets.append(annotations)
        predictions.append(detections)

    # Metrics
    map_metric = MeanAveragePrecision()
    map_result = map_metric.update(predictions, targets).compute()
    print(map_result)
    df = map_result.to_pandas()
    df.to_csv(args.out_dir / 'map_metric.csv', index=None)


if __name__ == '__main__':
    predict()
