import argparse
from pathlib import Path

import torch
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

# Map strings to the appropriate class definitions
MODEL_MAP = {
    'nano':   RFDETRNano,
    'small':  RFDETRSmall,
    'medium': RFDETRMedium,
    'base':   RFDETRBase,
    'large':  RFDETRLarge,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune RF-DETR on a Custom COCO Dataset')

    # Required arguments
    parser.add_argument(
        '--dataset-dir', '-ds',
        type=Path,
        required=True,
        help='Path to the root of your dataset (containing train/valid folders)'
    )

    # Model configuration
    parser.add_argument(
        '--model-size',
        type=str,
        default='medium',
        choices=MODEL_MAP.keys(),
        help='Which RF-DETR model size to fine-tune'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1,
        help='Number of classes in dataset (e.g., 1 for "conveyor_object")'
    )

    # Hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-bs',
        type=int,
        default=4,
        help='Batch size per GPU. Keep (batch_size * grad_accum_steps) around 16 for optimal stability.'
    )
    parser.add_argument(
        '--grad-accum-steps',
        type=int,
        default=4,
        help='Number of gradient accumulation steps.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    # Logging and Output
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='checkpoints/default',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Training device (cuda or cpu)'
    )

    return parser.parse_args()


def train():
    args = parse_args()
    ModelClass = MODEL_MAP[args.model_size]
    print(f'Loading pre-trained {args.model_size.upper()} model for {args.num_classes} class(es)...')
    model = ModelClass(num_classes=args.num_classes)

    print('Starting training job...')
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        progress_bar='tqdm'
    )

    print(f'Training complete. Weights and logs saved to: {args.output_dir}')


if __name__ == '__main__':
    train()
