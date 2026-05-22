from typing import Optional
from PIL import Image


class MathOCRHandler:
    """ocr pour les équations, convertit une image en latex via pix2tex"""

    def __init__(self, device: str = "cuda"):
        try:
            from pix2tex.cli import LatexOCR
        except Exception as e:
            raise ImportError(
                "pix2tex pas trouvé, installe le avec: pip install pix2tex"
            ) from e

        self.device = device
        self.model = LatexOCR()

        try:
            if hasattr(self.model, "model") and self.model.model is not None:
                self.model.model.to(self.device)
        except Exception:
            pass

    def image_to_latex(self, image: Image.Image) -> Optional[str]:
        try:
            latex = self.model(image)
            if latex:
                return latex.strip()
            return None
        except Exception:
            return None
