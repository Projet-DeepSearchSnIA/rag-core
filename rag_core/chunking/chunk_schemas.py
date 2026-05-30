"""
Schémas typés pour les chunks de document.

Avant : DocumentChunk avec metadata: dict ouvert → chaque consommateur
        (text_splitter, chunk_optimizer, pinecone_handler) accédait via
        chunk_meta.get('key', default) avec un défaut différent partout.
        Risque de divergence silencieuse difficile à détecter.

Après : DocumentChunk et ChunkMetadata sont des dataclasses. Les champs
        optionnels sont explicitement Optional[...] = None (le défaut est
        une déclaration de type, pas une valeur arbitraire). Tous les
        consommateurs partagent la même définition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class ChunkMetadata:
    """Métadonnées portées par chaque chunk.

    Tous les champs sont Optional ou ont un défaut typé (False/0/[]) qui
    correspond à l'absence de cette information dans le document source.
    """
    document_title: Optional[str] = None
    document_author: Optional[str] = None
    publication_id: Optional[int] = None
    attachment_id: Optional[int] = None
    user_id: Optional[int] = None
    is_public: bool = False

    extraction_method: Optional[str] = None
    page_has_images: bool = False
    has_tables: bool = False

    has_images: bool = False
    has_formulas: bool = False
    images: List[Dict] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    formulas: List[Dict] = field(default_factory=list)

    section_title: Optional[str] = None

    merged_from: Optional[List[str]] = None
    split_from: Optional[str] = None
    sub_index: Optional[int] = None

    def to_dict(self) -> Dict:
        """Sérialise en dict — None et listes vides sont préservés tels quels."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "ChunkMetadata":
        """Construit depuis un dict (JSON chargé). Les clés inconnues sont ignorées."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


@dataclass
class DocumentChunk:
    """Chunk de document avec ses métadonnées typées.

    char_count et word_count sont dérivés du contenu et calculés à la création.
    """
    chunk_id: str
    content: str
    document_id: str
    document_name: str
    page_numbers: List[int]
    chunk_index: int
    total_chunks: int
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    char_count: int = field(init=False)
    word_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.content)
        self.word_count = len(self.content.split())

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_numbers": self.page_numbers,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DocumentChunk":
        """Reconstruit un DocumentChunk depuis un dict JSON.

        Échoue explicitement si une clé requise est absente (pas de défaut
        silencieux). char_count et word_count sont recalculés depuis le contenu.
        page_numbers accepte aussi une string "1,2,3" pour la rétro-compat
        avec des chunks JSON historiques.
        """
        required = ("chunk_id", "content", "document_id", "document_name",
                    "page_numbers", "chunk_index", "total_chunks")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"DocumentChunk.from_dict : clés manquantes {missing}")

        raw_pages = data["page_numbers"]
        if isinstance(raw_pages, str):
            try:
                page_numbers = [int(p.strip()) for p in raw_pages.split(",") if p.strip()]
            except ValueError:
                page_numbers = []
        else:
            page_numbers = list(raw_pages)

        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            document_id=data["document_id"],
            document_name=data["document_name"],
            page_numbers=page_numbers,
            chunk_index=data["chunk_index"],
            total_chunks=data["total_chunks"],
            metadata=ChunkMetadata.from_dict(data.get("metadata")),
        )

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (f"DocumentChunk({self.chunk_id}, pages={self.page_numbers}, "
                f"words={self.word_count}, preview='{preview}')")
