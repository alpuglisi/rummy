"""Canonical playing-card vocabulary shared by every script in ``vision/``.

Index convention.  It deliberately matches the C++ rummy engine (``rummy_env.h``)
so a detected card can be handed straight to the RL agent without remapping::

    card_id    = suit_index * 13 + rank_index
    rank_index = 0 -> Ace, 1..8 -> 2..9, 9 -> 10, 10 -> Jack, 11 -> Queen, 12 -> King
    suit_index = 0 -> Clubs, 1 -> Diamonds, 2 -> Hearts, 3 -> Spades

Class names are ``<RANK><SUIT>``: ``AC``, ``10H``, ``QS`` ...  ``CARD_CLASSES[i]`` is
the name of engine card ``i``.

Public datasets use many naming schemes ("10C", "Ac", "ace of spades",
"ace_of_spades", Dutch "h10"/"kb", separate "hearts"/"king" objects ...).
``normalize_class_name`` folds all of them onto this vocabulary.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- vocabulary
RANKS: Tuple[str, ...] = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS: Tuple[str, ...] = ("C", "D", "H", "S")

RANK_WORDS: Dict[str, str] = {"A": "ace", "J": "jack", "Q": "queen", "K": "king"}
RANK_WORDS.update({r: r for r in RANKS[1:10]})
SUIT_WORDS: Dict[str, str] = {"C": "clubs", "D": "diamonds", "H": "hearts", "S": "spades"}

#: 52 canonical names, index == engine card id (suit-major, Ace first).
CARD_CLASSES: Tuple[str, ...] = tuple(f"{r}{s}" for s in SUITS for r in RANKS)
NUM_CARD_CLASSES: int = len(CARD_CLASSES)  # 52

JOKER = "JOKER"
PILE_FACE_DOWN = "PILE_FACE_DOWN"
PILE_FACE_UP = "PILE_FACE_UP"
CARD_BACK = "CARD_BACK"
#: Generic single-class label used by the stage-1 localizer ("there is a card here").
CARD = "CARD"

EXTRA_CLASSES: Tuple[str, ...] = (JOKER, PILE_FACE_DOWN, PILE_FACE_UP, CARD_BACK)
#: 52 cards followed by the extra game-state classes (ids 52..55).
ALL_CLASSES: Tuple[str, ...] = CARD_CLASSES + EXTRA_CLASSES

RANK_CLASSES: Tuple[str, ...] = tuple(f"RANK_{r}" for r in RANKS)
SUIT_CLASSES: Tuple[str, ...] = tuple(f"SUIT_{s}" for s in SUITS)
#: 17 classes for datasets that annotate rank and suit as separate objects.
PART_CLASSES: Tuple[str, ...] = RANK_CLASSES + SUIT_CLASSES

CLASS_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(ALL_CLASSES)}
PART_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(PART_CLASSES)}

#: Names of the label spaces a dataset can live in (see ``canonical_id``).
CLASS_SPACES: Tuple[str, ...] = ("cards52", "all", "parts", "card")


# --------------------------------------------------------------------------- card helpers
def card_id(rank: str, suit: str) -> int:
    """Engine id for a rank/suit pair, e.g. ``card_id("A", "C") == 0``."""
    return SUITS.index(suit.upper()) * 13 + RANKS.index(rank.upper())


def card_name(cid: int) -> str:
    if not 0 <= cid < NUM_CARD_CLASSES:
        raise ValueError(f"card id out of range: {cid}")
    return CARD_CLASSES[cid]


def parse_card(name: str) -> Tuple[str, str]:
    """``"10H" -> ("10", "H")``.  Raises ``ValueError`` for anything not in ``CARD_CLASSES``."""
    name = name.strip().upper()
    if name not in CARD_CLASSES:
        raise ValueError(f"not a canonical card name: {name!r}")
    return name[:-1], name[-1]


def rank_index(rank: str) -> int:
    return RANKS.index(rank.upper())


def suit_index(suit: str) -> int:
    return SUITS.index(suit.upper())


def rank_of(cid: int) -> str:
    return RANKS[cid % 13]


def suit_of(cid: int) -> str:
    return SUITS[cid // 13]


def is_red(name: str) -> bool:
    return parse_card(name)[1] in ("D", "H")


def card_asset_filename(name: str) -> str:
    """File name of the public-domain card art, e.g. ``"10_of_clubs.png"``."""
    r, s = parse_card(name)
    return f"{RANK_WORDS[r]}_of_{SUIT_WORDS[s]}.png"


def points_500_rummy(name: str) -> int:
    """Point value used by the engine: Ace 15, 10/J/Q/K 10, others 5."""
    r, _ = parse_card(name)
    if r == "A":
        return 15
    if r in ("10", "J", "Q", "K"):
        return 10
    return 5


# --------------------------------------------------------------------------- name normalisation
_RANK_TOKENS: Dict[str, str] = {
    "a": "A", "ace": "A", "1": "A", "one": "A",
    "2": "2", "two": "2", "3": "3", "three": "3", "4": "4", "four": "4",
    "5": "5", "five": "5", "6": "6", "six": "6", "7": "7", "seven": "7",
    "8": "8", "eight": "8", "9": "9", "nine": "9", "10": "10", "ten": "10", "t": "10",
    "j": "J", "jack": "J", "q": "Q", "queen": "Q", "k": "K", "king": "K",
}
_SUIT_TOKENS: Dict[str, str] = {
    "c": "C", "club": "C", "clubs": "C",
    "d": "D", "diamond": "D", "diamonds": "D",
    "h": "H", "heart": "H", "hearts": "H",
    "s": "S", "spade": "S", "spades": "S",
}
# pcc series (Deep Learning for Image and Video Processing) uses Dutch letters:
#   harten=h (hearts) klaveren=k (clubs) ruiten=r (diamonds) schoppen=s (spades)
#   aas=a (Ace) boer=b (Jack) vrouw=v (Queen) heer=h (King)   j = joker
_DUTCH_SUITS: Dict[str, str] = {"h": "H", "k": "C", "r": "D", "s": "S"}
_DUTCH_RANKS: Dict[str, str] = {"a": "A", "b": "J", "v": "Q", "h": "K"}
_DUTCH_RANKS.update({r: r for r in RANKS[1:10]})

_ENGLISH_COMPACT = re.compile(r"^(10|[2-9]|[ajqkt])([cdhs])$")       # 10C  As  Th
_ENGLISH_COMPACT_REV = re.compile(r"^([cdhs])(10|[2-9]|[ajqk])$")    # C10  sA
_DUTCH_COMPACT = re.compile(r"^([hkrs])(10|[2-9]|[abhv])$")          # h10  kb  sv

#: Naming styles understood by ``normalize_class_name``.
NAME_STYLES: Tuple[str, ...] = ("english", "pcc_dutch")


def _clean(raw: str) -> str:
    return re.sub(r"[\s\-_]+", " ", raw.strip().lower()).strip()


def normalize_class_name(raw: str, style: str = "english") -> Optional[str]:
    """Map a dataset's raw class name onto the canonical vocabulary.

    Returns one of ``CARD_CLASSES``, ``EXTRA_CLASSES``, ``PART_CLASSES`` (for
    datasets that label rank and suit separately), ``CARD`` (generic card) or
    ``None`` when the name is not understood.

    ``style="english"`` (default) understands ``10C``, ``Ac``, ``Th``, ``C10``,
    ``ace of spades``, ``ace_of_spades``, ``king-hearts``, ``hearts king``,
    ``red_joker``, ``pile-face-down``, ``card back`` ...
    ``style="pcc_dutch"`` understands the Dutch scheme of the pcc datasets
    (``h10``, ``ka``, ``sv``, ``j``).  The two compact schemes overlap (``kh`` is
    King of Hearts in English but King of Clubs in Dutch) so the style must be
    chosen per dataset; it is recorded in ``config.DATASETS``.
    """
    if style not in NAME_STYLES:
        raise ValueError(f"unknown name style {style!r}; expected one of {NAME_STYLES}")
    s = _clean(raw)
    if not s:
        return None
    compact = s.replace(" ", "")

    # game-state / special classes (shared by both styles)
    if "joker" in compact:
        return JOKER
    if "pile" in compact and "down" in compact:
        return PILE_FACE_DOWN
    if "pile" in compact and "up" in compact:
        return PILE_FACE_UP
    if compact in {"back", "cardback", "backofcard", "cardsback", "facedown", "backside"}:
        return CARD_BACK
    if compact in {"card", "cards", "playingcard", "playingcards", "cardface", "faceup"}:
        return CARD

    if style == "pcc_dutch":
        if compact == "j":
            return JOKER
        m = _DUTCH_COMPACT.match(compact)
        if m:
            return f"{_DUTCH_RANKS[m.group(2)]}{_DUTCH_SUITS[m.group(1)]}"
        return None

    m = _ENGLISH_COMPACT.match(compact)
    if m:
        return f"{_RANK_TOKENS[m.group(1)]}{m.group(2).upper()}"
    m = _ENGLISH_COMPACT_REV.match(compact)
    if m:
        return f"{_RANK_TOKENS[m.group(2)]}{m.group(1).upper()}"

    words = [w for w in s.split(" ") if w not in ("of", "the")]
    ranks = [_RANK_TOKENS[w] for w in words if w in _RANK_TOKENS]
    suits = [_SUIT_TOKENS[w] for w in words if w in _SUIT_TOKENS]
    if len(ranks) == 1 and len(suits) == 1:
        return f"{ranks[0]}{suits[0]}"
    if len(ranks) == 1 and not suits:
        return f"RANK_{ranks[0]}"
    if len(suits) == 1 and not ranks:
        return f"SUIT_{suits[0]}"
    return None


def canonical_id(canon: Optional[str], space: str = "cards52") -> Optional[int]:
    """Id of a canonical name inside a label space, or ``None`` if it does not belong.

    * ``cards52``: the 52 cards only (the main detector / classifier space)
    * ``all``: 52 cards + ``EXTRA_CLASSES`` (56 ids; used by the pcc pile classes)
    * ``parts``: the 17 ``PART_CLASSES`` (rank / suit annotated separately)
    * ``card``: single generic class -> id 0 for any card, part or ``CARD`` label
    """
    if canon is None:
        return None
    if space == "cards52":
        return CLASS_TO_ID[canon] if canon in CARD_CLASSES else None
    if space == "all":
        return CLASS_TO_ID.get(canon)
    if space == "parts":
        return PART_TO_ID.get(canon)
    if space == "card":
        if canon in CARD_CLASSES or canon in PART_CLASSES or canon in (CARD, JOKER):
            return 0
        return None
    raise ValueError(f"unknown class space {space!r}; expected one of {CLASS_SPACES}")


def class_names_for_space(space: str) -> List[str]:
    if space == "cards52":
        return list(CARD_CLASSES)
    if space == "all":
        return list(ALL_CLASSES)
    if space == "parts":
        return list(PART_CLASSES)
    if space == "card":
        return [CARD]
    raise ValueError(f"unknown class space {space!r}; expected one of {CLASS_SPACES}")


def map_dataset_names(names: Iterable[str], style: str = "english", space: str = "cards52") -> Dict[int, Optional[int]]:
    """Build ``{raw_class_index: canonical_id_or_None}`` for a dataset's ``names`` list.

    ``None`` means "drop boxes of this class".  Works for both ``list`` names
    (index = position) and ``dict`` names (``{id: name}``) as found in YOLO
    ``data.yaml`` files.
    """
    if isinstance(names, dict):
        items = [(int(k), str(v)) for k, v in names.items()]
    else:
        items = list(enumerate(str(n) for n in names))
    return {idx: canonical_id(normalize_class_name(raw, style), space) for idx, raw in items}


def describe_mapping(names: Iterable[str], style: str = "english", space: str = "cards52") -> List[Tuple[int, str, Optional[str], Optional[int]]]:
    """Human-readable rows ``(raw_idx, raw_name, canonical_name, canonical_id)`` for logging."""
    if isinstance(names, dict):
        items = [(int(k), str(v)) for k, v in names.items()]
    else:
        items = list(enumerate(str(n) for n in names))
    rows = []
    for idx, raw in items:
        canon = normalize_class_name(raw, style)
        rows.append((idx, raw, canon, canonical_id(canon, space)))
    return rows
