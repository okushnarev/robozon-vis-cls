import json
import logging
from pathlib import Path
from typing import Literal, Type, Dict

from rfdetr import RFDETR

logger = logging.getLogger(__name__)

ModelMode = Literal['detection', 'segmentation']


def _get_model_class(mode: ModelMode, model_name: str) -> Type[RFDETR]:
    if mode == 'detection':
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
        model_map: Dict[str, Type[RFDETR]] = {
            'RFDETRNano':   RFDETRNano,
            'RFDETRSmall':  RFDETRSmall,
            'RFDETRMedium': RFDETRMedium,
            'RFDETRLarge':  RFDETRLarge,
        }
    elif mode == 'segmentation':
        from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge
        model_map = {
            'RFDETRSegNano':   RFDETRSegNano,
            'RFDETRSegSmall':  RFDETRSegSmall,
            'RFDETRSegMedium': RFDETRSegMedium,
            'RFDETRSegLarge':  RFDETRSegLarge,
        }
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from: ['detection', 'segmentation']")

    if model_name not in model_map:
        raise KeyError(
            f"Model '{model_name}' not found for mode '{mode}'. "
            f"Available models: {list(model_map.keys())}"
        )

    return model_map[model_name]


def load_model(
        ckpt_dir: Path,
        ckpt_type: str,
        mode: ModelMode,
) -> RFDETR:
    config_path = ckpt_dir / 'training_config.json'
    if '.' in ckpt_type:
        weights_path = ckpt_dir / ckpt_type
    else:
        weights_path = ckpt_dir / f'{ckpt_type}.pth'

    if not config_path.exists():
        raise FileNotFoundError(f'Configuration file not found at: {config_path}')

    if not weights_path.exists():
        raise FileNotFoundError(f'Weights file not found at: {weights_path}')

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_name = config['model_config']['model_name']
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f'Failed to parse model configuration from {config_path}: {e}') from e

    # Retrieve class and instantiate
    ModelClass = _get_model_class(mode, model_name)

    logger.info(f'Loading {model_name} in {mode} mode.')
    model = ModelClass(pretrain_weights=weights_path)

    # Optimize for inference if applicable
    if hasattr(model, 'optimize_for_inference'):
        model.optimize_for_inference()

    return model