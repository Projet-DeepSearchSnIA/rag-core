"""
Tests fumée pour le CLI unifié scripts/rag.py.

Vérifie que :
  - le module CLI s'importe sans erreur
  - chaque sous-commande accepte --help
  - PDFExtractor s'initialise correctement à partir de configs/baseline.yaml
    (régression : le précédent index.py instanciait PDFExtractor() sans config)
  - ExtractedDocument.from_dict reconstruit fidèlement depuis to_dict
"""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import load_baseline, make_doc
from rag_core.extraction.pdf_extractor import PDFExtractor
from rag_core.extraction.document_schemas import ExtractedDocument


ROOT = Path(__file__).parent.parent
SUBCOMMANDS = ("extract", "chunk", "upload", "index", "retrieve", "ask")


def test_scripts_rag_import_sans_erreur():
    importlib.import_module("scripts.rag")


def test_scripts_rag_help():
    result = subprocess.run(
        [sys.executable, "scripts/rag.py", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--help a renvoyé {result.returncode}\n{result.stderr}"
    for sub in SUBCOMMANDS:
        assert sub in result.stdout, f"sous-commande {sub} absente de --help"


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_chaque_sous_commande_a_son_help(subcommand):
    result = subprocess.run(
        [sys.executable, "scripts/rag.py", subcommand, "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"{subcommand} --help a renvoyé {result.returncode}\n{result.stderr}"
    assert "--config" in result.stdout, f"--config absent du help de {subcommand}"


def test_sous_commande_manquante_echec_propre():
    """Sans sous-commande, rag.py doit afficher l'aide et échouer (code != 0)."""
    result = subprocess.run(
        [sys.executable, "scripts/rag.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0


def test_baseline_yaml_construit_pdfextractor_sans_erreur():
    """Régression : PDFExtractor doit s'initialiser avec la section extraction de baseline.yaml."""
    cfg = load_baseline()
    assert "extraction" in cfg, "section [extraction] manquante de baseline.yaml"
    extractor = PDFExtractor(config=cfg["extraction"])
    assert extractor.use_math_ocr is not None
    assert extractor.extract_images is not None


def test_baseline_yaml_contient_toutes_les_sections_attendues():
    """baseline.yaml doit être la source unique de vérité pour le pipeline."""
    cfg = load_baseline()
    for section in ("extraction", "chunking", "embedding", "vectorstore", "retrieval", "generation"):
        assert section in cfg, f"section [{section}] manquante de baseline.yaml"


def test_extracted_document_roundtrip_to_from_dict():
    """from_dict doit être l'inverse de to_dict pour permettre la sous-commande chunk."""
    doc = make_doc(["page 1 texte", "page 2 texte"])
    doc.metadata.title = "Test"
    doc.metadata.publication_id = 42

    roundtrip = ExtractedDocument.from_dict(doc.to_dict())

    assert roundtrip.document_id == doc.document_id
    assert roundtrip.filename == doc.filename
    assert len(roundtrip.pages) == 2
    assert roundtrip.pages[0].page_number == 1
    assert roundtrip.pages[0].content_blocks[0].content == "page 1 texte"
    assert roundtrip.metadata.title == "Test"
    assert roundtrip.metadata.publication_id == 42
