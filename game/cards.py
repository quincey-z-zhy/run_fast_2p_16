import random

import sys
import os
path_root = os.path.dirname(os.path.dirname(__file__))
if path_root not in sys.path:
    sys.path.append(path_root)
path_current = os.path.dirname(__file__)
if path_current in sys.path:
    sys.path.remove(path_current)

from game.config import DECK_COUNTS, HAND_SIZE, CARD_TO_INT, CARD_TO_CHAR


def shuffle() -> tuple[list[int], list[int]]:
    '''
    洗牌发牌

    从 45 张牌中各发 15 张给两位玩家，剩余 15 张为死牌（不参与游戏）。

    :return p0_cards: 玩家 0 的手牌
    :return p1_cards: 玩家 1 的手牌
    '''

    deck = []
    for face, count in DECK_COUNTS.items():
        deck.extend([face] * count)
    random.shuffle(deck)

    p0_cards = sorted(deck[:HAND_SIZE])
    p1_cards = sorted(deck[HAND_SIZE: 2 * HAND_SIZE])

    return p0_cards, p1_cards


def cards_to_str(cards: list[int]) -> str:
    '''
    将牌值列表转化为字符串格式

    :param cards: 牌值列表
    :return: 牌面字符串
    '''

    return ''.join(CARD_TO_CHAR[c] for c in cards)


def cards_to_int(s: str) -> list[int]:
    '''
    将牌面字符串转化为牌值列表

    :param s: 牌面字符串
    :return: 升序牌值列表
    '''

    return sorted(CARD_TO_INT[c] for c in s)
