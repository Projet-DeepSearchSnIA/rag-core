"""
Tests pour TextPreprocessor — nettoyage et normalisation du texte extrait.

Le preprocesseur tourne juste après l'extraction PDF, avant le chunking.
On teste chaque règle de nettoyage isolément pour pouvoir débugger rapidement
quand quelque chose casse. Pas besoin de vrai PDF ici, on fabrique des blocs à la main.
"""
from rag_core.extraction.preprocessor import TextPreprocessor
from rag_core.extraction.document_schemas import ContentBlock


def _bloc(type_: str, content: str, page: int = 1) -> ContentBlock:
    """Raccourci pour créer un ContentBlock sans se répéter dans chaque test."""
    return ContentBlock(type=type_, content=content, page_number=page)


def test_blocs_vides_supprimes():
    """Les blocs sans contenu ou avec que des espaces doivent disparaître."""
    preprocesseur = TextPreprocessor()
    blocs = [
        _bloc("text", ""),
        _bloc("text", "   \n  "),
        _bloc("text", "Un vrai contenu."),
    ]
    resultat = preprocesseur.preprocess_blocks(blocs)
    assert len(resultat) == 1
    assert resultat[0].content == "Un vrai contenu."


def test_normalisation_espaces_multiples():
    """Plusieurs espaces consécutifs doivent devenir un seul espace."""
    preprocesseur = TextPreprocessor()
    bloc = _bloc("text", "Bonjour   monde,  comment   ça va ?")
    resultat = preprocesseur.preprocess_blocks([bloc])
    assert "  " not in resultat[0].content


def test_trait_union_fin_de_ligne_colle_les_mots():
    """Un mot coupé par un tiret en fin de ligne doit être recollé.

    Cas typique des PDFs : "ap-\nprentissage" → "apprentissage".
    """
    preprocesseur = TextPreprocessor()
    bloc = _bloc("text", "ap-\nprentissage automatique")
    resultat = preprocesseur.preprocess_blocks([bloc])
    assert "apprentissage" in resultat[0].content


def test_formule_latex_non_touchee():
    """Le contenu d'un bloc 'formula' ne doit JAMAIS être modifié.

    Les transformations regex casseraient le LaTeX — on ne touche pas à ce type.
    """
    preprocesseur = TextPreprocessor()
    latex_brut = r"\frac{d}{dx}\sin(x) = \cos(x)"
    bloc = _bloc("formula", latex_brut)
    resultat = preprocesseur.preprocess_blocks([bloc])
    assert resultat[0].content == latex_brut


def test_guillemets_typographiques_remplaces():
    """Les guillemets courbes doivent devenir des guillemets droits.

    Utile pour uniformiser les embeddings — les modèles voient ' et ' comme différents.
    """
    preprocesseur = TextPreprocessor()
    bloc = _bloc("text", "l’exemple et ‘test’")
    resultat = preprocesseur.preprocess_blocks([bloc])
    # les guillemets courbes doivent avoir disparu
    assert "‘" not in resultat[0].content
    assert "’" not in resultat[0].content


def test_tirets_longs_remplaces():
    """Les tirets longs (em-dash, en-dash) deviennent des tirets simples."""
    preprocesseur = TextPreprocessor()
    bloc = _bloc("text", "résultat – excellent — vraiment")
    resultat = preprocesseur.preprocess_blocks([bloc])
    assert "–" not in resultat[0].content
    assert "—" not in resultat[0].content
    assert "-" in resultat[0].content


def test_urls_extraites_dans_metadata():
    """Les URLs trouvées dans le texte doivent apparaître dans block.metadata['urls']."""
    preprocesseur = TextPreprocessor()
    bloc = _bloc("text", "Voir https://example.com pour plus d'infos.")
    resultat = preprocesseur.preprocess_blocks([bloc])
    assert "urls" in resultat[0].metadata
    assert any("example.com" in url for url in resultat[0].metadata["urls"])


def test_emails_extraits_dans_metadata():
    """Les adresses email dans le texte doivent apparaître dans block.metadata['emails']."""
    preprocesseur = TextPreprocessor()
    bloc = _bloc("text", "Contactez auteur@labo.fr pour toute question.")
    resultat = preprocesseur.preprocess_blocks([bloc])
    assert "emails" in resultat[0].metadata
    assert "auteur@labo.fr" in resultat[0].metadata["emails"]


def test_patrons_repetes_supprimes():
    """Un texte répété plus de 3 fois sur différentes pages est considéré comme header/footer.

    C'est la logique de _detect_repeated_patterns — on vérifie qu'elle élimine bien ces blocs.
    """
    preprocesseur = TextPreprocessor()
    # le même texte court 4 fois → détecté comme répétitif
    entete = "Confidentiel — Ne pas diffuser"
    blocs = [_bloc("text", entete, page=i) for i in range(1, 5)]
    blocs.append(_bloc("text", "Du vrai contenu scientifique.", page=5))
    resultat = preprocesseur.preprocess_blocks(blocs)
    contenus = [b.content for b in resultat]
    # l'entête répété doit avoir disparu
    assert entete not in contenus
    assert any("vrai contenu" in c for c in contenus)


def test_tableau_lignes_vides_supprimees():
    """Pour un bloc 'table', les lignes vides internes doivent être supprimées."""
    preprocesseur = TextPreprocessor()
    bloc = _bloc("table", "col1 | col2\n\ncell1 | cell2\n\n")
    resultat = preprocesseur.preprocess_blocks([bloc])
    # aucune ligne vide ne doit subsister
    for ligne in resultat[0].content.split("\n"):
        assert ligne.strip() != ""


def test_sans_config_les_defauts_sont_actifs():
    """Sans config explicite, toutes les règles de nettoyage sont actives par défaut."""
    preprocesseur = TextPreprocessor()
    assert preprocesseur.normalize_whitespace is True
    assert preprocesseur.merge_hyphenated is True
    assert preprocesseur.extract_urls is True
    assert preprocesseur.extract_emails is True
    assert preprocesseur.remove_headers_footers is True


def test_config_desactive_extraction_urls():
    """On peut désactiver l'extraction d'URLs via la config."""
    preprocesseur = TextPreprocessor(config={"extract_urls": False})
    bloc = _bloc("text", "Voir https://example.com pour plus d'infos.")
    resultat = preprocesseur.preprocess_blocks([bloc])
    # pas de clé 'urls' si désactivé
    assert "urls" not in (resultat[0].metadata or {})


def test_plusieurs_blocs_independants():
    """Plusieurs blocs valides sont tous retournés, chacun nettoyé séparément."""
    preprocesseur = TextPreprocessor()
    blocs = [
        _bloc("text", "Premier paragraphe."),
        _bloc("text", "Deuxième   paragraphe."),
        _bloc("title", "Titre de section"),
    ]
    resultat = preprocesseur.preprocess_blocks(blocs)
    assert len(resultat) == 3


def test_bloc_trop_court_apres_nettoyage_supprime():
    """Un bloc qui ne contient que des caractères spéciaux vides après nettoyage est supprimé."""
    preprocesseur = TextPreprocessor()
    # caractères de contrôle qui seront supprimés par _clean_special_chars
    bloc = _bloc("text", "\x00\x01\x02\x03")
    resultat = preprocesseur.preprocess_blocks([bloc])
    assert resultat == []