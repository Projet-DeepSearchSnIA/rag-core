from rag_core.retrieval.retriever import PineconeRetriever


def test_truncate_pour_rerank_court():
    # on teste la méthode privée directement pour valider le ratio 4 chars/token
    text = "a" * 50
    result = PineconeRetriever._truncate_for_rerank(None, text, max_tokens=200)
    assert result == text


def test_truncate_pour_rerank_long():
    text = "x" * 1000
    result = PineconeRetriever._truncate_for_rerank(None, text, max_tokens=200)
    assert len(result) == 800  # 200 tokens * 4 chars


def test_truncate_pour_rerank_vide():
    result = PineconeRetriever._truncate_for_rerank(None, "", max_tokens=200)
    assert result == ""


def test_parse_json_field_liste():
    r = PineconeRetriever._parse_json_field(None, '["a", "b"]')
    assert r == ["a", "b"]


def test_parse_json_field_non_json():
    r = PineconeRetriever._parse_json_field(None, "simple texte")
    assert r == "simple texte"


def test_parse_json_field_deja_liste():
    r = PineconeRetriever._parse_json_field(None, ["a", "b"])
    assert r == ["a", "b"]


def test_parse_list_field_csv():
    r = PineconeRetriever._parse_list_field(None, "page1,page2,page3")
    assert r == ["page1", "page2", "page3"]


def test_parse_list_field_liste():
    r = PineconeRetriever._parse_list_field(None, ["x", "y"])
    assert r == ["x", "y"]


def test_parse_list_field_vide():
    r = PineconeRetriever._parse_list_field(None, "")
    assert r == []
