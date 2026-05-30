"""
Tests fumée pour les scripts CLI.

Garantit que `scripts/index.py` et `scripts/query.py` :
  - peuvent s'importer sans crasher
  - acceptent --help (argparse correctement défini)
  - construisent PDFExtractor avec la section extraction de baseline.yaml
    sans lever ValueError (régression : l'ancien index.py instanciait
    PDFExtractor() sans config et plantait au runtime).
"""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import load_baseline
from rag_core.extraction.pdf_extractor import PDFExtractor


ROOT = Path(__file__).parent.parent


def test_scripts_index_import_sans_erreur():
    importlib.import_module("scripts.index")


def test_scripts_query_import_sans_erreur():
    importlib.import_module("scripts.query")


def test_scripts_index_help():
    result = subprocess.run(
        [sys.executable, "scripts/index.py", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--help a renvoyé {result.returncode}\n{result.stderr}"
    assert "--index" in result.stdout
    assert "--namespace" in result.stdout
    assert "--config" in result.stdout


def test_scripts_query_help():
    result = subprocess.run(
        [sys.executable, "scripts/query.py", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"--help a renvoyé {result.returncode}\n{result.stderr}"
    assert "question" in result.stdout


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
