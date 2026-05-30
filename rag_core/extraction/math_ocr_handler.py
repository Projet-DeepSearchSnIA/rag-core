import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class MathOCRHandler:
    """OCR pour équations mathématiques — convertit une image en LaTeX.

    Utilise VisionEncoderDecoderModel via transformers, sans dépendance sur
    pix2tex ni timm. Le modèle est téléchargé automatiquement depuis HuggingFace
    au premier appel.
    """

    def __init__(self, model_id: str, model_subfolder: str, device: str):
        try:
            import torch
            from transformers import AutoTokenizer, VisionEncoderDecoderModel
            from transformers import ViTImageProcessor
        except ImportError as e:
            raise ImportError(
                "transformers et torch sont requis pour MathOCRHandler"
            ) from e

        self.device = device
        logger.info("chargement du modèle math OCR depuis %s/%s", model_id, model_subfolder)

        self._processor = ViTImageProcessor.from_pretrained(model_id, subfolder=model_subfolder)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=model_subfolder)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_id, subfolder=model_subfolder)
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        logger.info("modèle math OCR chargé sur %s", device)

    def image_to_latex(self, image: Image.Image) -> Optional[str]:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            pixel_values = self._processor(
                image, return_tensors="pt"
            ).pixel_values.to(self.device)

            with self._torch.no_grad():
                generated_ids = self._model.generate(pixel_values)

            latex = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return latex.strip() if latex.strip() else None
        except Exception as e:
            logger.warning("erreur lors de la conversion image→LaTeX: %s", e)
            return None