import pytest

from vision import cards


def test_engine_index_convention():
    # suit-major, Ace first: matches rummy_env.h (get_suit = id / 13, get_rank = id % 13, rank 0 = Ace)
    assert cards.CARD_CLASSES[0] == "AC"
    assert cards.CARD_CLASSES[12] == "KC"
    assert cards.CARD_CLASSES[13] == "AD"
    assert cards.CARD_CLASSES[51] == "KS"
    assert len(cards.CARD_CLASSES) == 52 == len(set(cards.CARD_CLASSES))
    for cid, name in enumerate(cards.CARD_CLASSES):
        r, s = cards.parse_card(name)
        assert cards.card_id(r, s) == cid
        assert cards.rank_of(cid) == r and cards.suit_of(cid) == s
        assert cards.card_name(cid) == name


def test_points_match_engine():
    assert cards.points_500_rummy("AS") == 15
    assert cards.points_500_rummy("10H") == 10 == cards.points_500_rummy("KD")
    assert cards.points_500_rummy("2C") == 5 == cards.points_500_rummy("9S")


def test_asset_filename():
    assert cards.card_asset_filename("10C") == "10_of_clubs.png"
    assert cards.card_asset_filename("AS") == "ace_of_spades.png"
    assert cards.card_asset_filename("QH") == "queen_of_hearts.png"


@pytest.mark.parametrize("raw,expected", [
    ("10C", "10C"), ("AS", "AS"), ("Ac", "AC"), ("10h", "10H"), ("Th", "10H"), ("C10", "10C"), ("sA", "AS"),
    ("ace of spades", "AS"), ("ace_of_spades", "AS"), ("king-hearts", "KH"), ("hearts king", "KH"),
    ("ten of diamonds", "10D"), ("Queen Of Clubs", "QC"),
    ("joker", "JOKER"), ("red_joker", "JOKER"), ("Black Joker", "JOKER"),
    ("pile-face-down", "PILE_FACE_DOWN"), ("pile_face_up", "PILE_FACE_UP"), ("Pile Face Up", "PILE_FACE_UP"),
    ("back", "CARD_BACK"), ("card_back", "CARD_BACK"),
    ("card", "CARD"), ("playing card", "CARD"),
    ("hearts", "SUIT_H"), ("Clubs", "SUIT_C"), ("ace", "RANK_A"), ("10", "RANK_10"), ("K", "RANK_K"),
    ("banana", None), ("", None),
])
def test_normalize_english(raw, expected):
    assert cards.normalize_class_name(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("h10", "10H"), ("ha", "AH"), ("hb", "JH"), ("hv", "QH"), ("hh", "KH"),
    ("k2", "2C"), ("ka", "AC"), ("r9", "9D"), ("rb", "JD"), ("s10", "10S"), ("sv", "QS"),
    ("j", "JOKER"), ("pile-face-down", "PILE_FACE_DOWN"), ("pile-face-up", "PILE_FACE_UP"), ("x3", None),
])
def test_normalize_pcc_dutch(raw, expected):
    assert cards.normalize_class_name(raw, style="pcc_dutch") == expected


def test_pcc_full_class_list_maps_to_55_distinct():
    names = [f"{s}{r}" for s in "hkrs" for r in ["10", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "h", "v"]]
    names += ["j", "pile-face-down", "pile-face-up"]
    mapped = cards.map_dataset_names(names, style="pcc_dutch", space="all")
    assert len(mapped) == 55 and None not in mapped.values()
    assert len(set(mapped.values())) == 55
    m52 = cards.map_dataset_names(names, style="pcc_dutch", space="cards52")
    assert sum(v is None for v in m52.values()) == 3  # joker + 2 pile classes dropped


def test_jackfurby_class_file_and_ordering():
    from pathlib import Path
    from vision.config import REPO_ROOT
    txt = (REPO_ROOT / "vision" / "configs" / "jackfurby_card_classes.txt").read_text().split()
    pairs = dict(zip(txt[0::2], map(int, txt[1::2])))
    assert pairs["2C"] == 0 and pairs["AS"] == 51 and len(pairs) == 52
    # every JackFurby name normalises to itself and gets a canonical id
    for name in pairs:
        assert cards.normalize_class_name(name) == name
        assert cards.canonical_id(name) is not None


def test_canonical_id_spaces():
    assert cards.canonical_id("AC") == 0 and cards.canonical_id("KS") == 51
    assert cards.canonical_id("JOKER") is None
    assert cards.canonical_id("JOKER", "all") == 52 and cards.canonical_id("PILE_FACE_UP", "all") == 54
    assert cards.canonical_id("RANK_A", "parts") == 0 and cards.canonical_id("SUIT_S", "parts") == 16
    assert cards.canonical_id("AC", "parts") is None
    assert cards.canonical_id("AC", "card") == 0 and cards.canonical_id("SUIT_H", "card") == 0
    assert cards.canonical_id("PILE_FACE_UP", "card") is None
    with pytest.raises(ValueError):
        cards.canonical_id("AC", "nope")


def test_map_dataset_names_dict_and_list():
    m = cards.map_dataset_names({"0": "10C", "1": "joker", "2": "AS"})
    assert m == {0: cards.CLASS_TO_ID["10C"], 1: None, 2: cards.card_id("A", "S")}  # AS = 3*13+0 = 39
    m = cards.map_dataset_names(["ace of clubs", "banana"])
    assert m == {0: 0, 1: None}
