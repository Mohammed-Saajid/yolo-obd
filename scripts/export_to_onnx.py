import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from ultralytics import YOLO

# Configure logging for professional observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class YOLOExporter:
    """Handles the conversion of YOLO models to deployment-optimized formats."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Weights not found at: {self.model_path}")
        
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """
        Loads the YOLO model while safely handling PyTorch 2.6+ 
        weights_only restrictions for trusted local weights.
        """
        logger.info(f"Loading model: {self.model_path.name}")
        
        # Capture the original state to prevent side effects in larger systems
        original_load = torch.load
        
        try:
            # Inline patch to allow loading Ultralytics custom classes
            # This is scoped strictly to the model initialization
            torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
            model = YOLO(str(self.model_path))
            return model
        except Exception as e:
            logger.error(f"Critical failure during model instantiation: {e}")
            raise
        finally:
            # Restore original torch.load immediately
            torch.load = original_load

    def export_to_onnx(
        self, 
        output_path: str, 
        imgsz: int = 320, 
        opset: int = 20
    ) -> Path:
        """
        Exports the loaded model to ONNX with optimizations for TensorRT/Triton.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting ONNX export (imgsz={imgsz}, opset={opset})...")
        
        try:
            # Ultralytics export returns the path of the saved file
            exported_path = self.model.export(
                format='onnx',
                imgsz=imgsz,
                opset=opset,
                dynamic=True,
                simplify=True,
                verbose=False
            )
            logger.info(f"Export successful: {exported_path}")
            return Path(exported_path)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise RuntimeError(f"ONNX export process encountered an error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Professional YOLO26n to ONNX Export Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to input .pt file")
    parser.add_argument("--output", type=str, required=True, help="Path for output .onnx file")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size")
    parser.add_argument("--opset", type=int, default=20, help="ONNX opset version")

    args = parser.parse_args()

    try:
        exporter = YOLOExporter(args.weights)
        exporter.export_to_onnx(args.output, imgsz=args.imgsz, opset=args.opset)
    except Exception as e:
        logger.critical(f"Workflow terminated: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()