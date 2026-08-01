import json
from pathlib import Path
from typing import Literal

from rfdetr import RFDETR


class RFDETRModelWrapper:
    @staticmethod
    def get_model(
            ckpt_dir: Path,
            ckpt_type: str,
            mode: Literal['detection', 'segmentation'],
    ) -> RFDETR:

        if mode == 'detection':
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

            MODEL_MAP = {
                'RFDETRNano':   RFDETRNano,
                'RFDETRSmall':  RFDETRSmall,
                'RFDETRMedium': RFDETRMedium,
                'RFDETRLarge':  RFDETRLarge,
            }
        elif mode == 'segmentation':
            from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge
            MODEL_MAP = {
                'RFDETRSegNano':   RFDETRSegNano,
                'RFDETRSegSmall':  RFDETRSegSmall,
                'RFDETRSegMedium': RFDETRSegMedium,
                'RFDETRSegLarge':  RFDETRSegLarge,
            }
        else:
            raise ValueError(f"No mode '{mode}' available. Choose from: ['detection', 'segmentation']")

        with open(ckpt_dir / 'training_config.json', 'r') as f:
            config = json.load(f)

        ModelClass = MODEL_MAP[config['model_config']['model_name']]
        model = ModelClass(pretrain_weights=ckpt_dir / f'{ckpt_type}.pth')
        model.optimize_for_inference()

        return model
