import copy

import sys
import os
path_root = os.path.dirname(os.path.dirname(__file__))
if path_root not in sys.path:
    sys.path.append(path_root)
path_current = os.path.dirname(__file__)
if path_current in sys.path:
    sys.path.remove(path_current)

from game.config import (
    FACES, STRAIGHT_FACES, POCKET_TRIPLE_FACES, QUAD_FACES,
    FORMATIONS, FORMATION_NAMES,
    MIN_STRAIGHT_LEN, MAX_STRAIGHT_LEN,
    MIN_S_POCKET_LEN, MAX_S_POCKET_LEN,
    MIN_S_TRIPS_LEN, MAX_S_TRIPS_LEN,
)
from game.cards import cards_to_str


class Action(object):
    '''
    游戏动作
    '''

    def __init__(
            self,
            formation: int,
            start_card: int = None,
            end_card: int = None,
            tail_cards: list[int] = None):
        '''
        游戏动作，初始化

        :param formation: 牌型编码
        :param start_card: 主牌起始牌值（顺牌型为最低牌，非顺牌型为该牌面值）
        :param end_card: 主牌结束牌值（顺牌型使用，非顺牌型为 None）
        :param tail_cards: 带牌列表（已排序的牌值列表）
        '''

        self.formation = formation
        self.start_card = start_card
        self.end_card = end_card
        self.tail_cards = copy.deepcopy(tail_cards) if tail_cards else []

    @property
    def num_kicker(self) -> int:
        '''带牌数量'''
        return len(self.tail_cards)

    @property
    def length(self) -> int:
        '''顺牌型的节数（非顺牌型返回 1）'''
        if self.end_card is not None:
            return self.end_card - self.start_card + 1
        return 1

    @property
    def is_legal(self) -> bool:
        '''动作是否合法'''

        f = self.formation
        sc, ec, tc = self.start_card, self.end_card, self.tail_cards

        # 不出牌
        if f == FORMATIONS['none']:
            return sc is None and ec is None and not tc

        # 单牌
        elif f == FORMATIONS['single']:
            return sc in FACES and ec is None and not tc

        # 对子
        elif f == FORMATIONS['pocket']:
            return sc in POCKET_TRIPLE_FACES and ec is None and not tc

        # 连对
        elif f == FORMATIONS['s_pocket']:
            if sc not in POCKET_TRIPLE_FACES or ec not in POCKET_TRIPLE_FACES:
                return False
            length = ec - sc + 1
            if not (MIN_S_POCKET_LEN <= length <= MAX_S_POCKET_LEN):
                return False
            return ec is not None and not tc

        # 三条（带 0/1/2 张单牌）
        elif f == FORMATIONS['trips']:
            if sc not in POCKET_TRIPLE_FACES or ec is not None:
                return False
            if len(tc) > 2:
                return False
            # 带牌不能与主牌相同，且每张牌值最多 3 张
            face_counts = {}
            for c in tc:
                if c == sc:
                    return False
                face_counts[c] = face_counts.get(c, 0) + 1
                if face_counts[c] > 3:
                    return False
            return True

        # 飞机（连续三条带 0/N/2N 张单牌）
        elif f == FORMATIONS['s_trips']:
            if sc not in POCKET_TRIPLE_FACES or ec not in POCKET_TRIPLE_FACES:
                return False
            N = ec - sc + 1
            if not (MIN_S_TRIPS_LEN <= N <= MAX_S_TRIPS_LEN):
                return False
            if len(tc) not in (0, N, 2 * N):
                return False
            main_faces = set(range(sc, ec + 1))
            face_counts = {}
            for c in tc:
                if c in main_faces:
                    return False
                face_counts[c] = face_counts.get(c, 0) + 1
                if face_counts[c] > 3:
                    return False
            return True

        # 四带（带 0/1/2/3 张单牌，主牌牌面须在 QUAD_FACES）
        elif f == FORMATIONS['quads']:
            if sc not in QUAD_FACES or ec is not None:
                return False
            if len(tc) > 3:
                return False
            face_counts = {}
            for c in tc:
                if c == sc:
                    return False
                face_counts[c] = face_counts.get(c, 0) + 1
                if face_counts[c] > 3:
                    return False
            return True

        # 顺子
        elif f == FORMATIONS['straight']:
            if sc not in STRAIGHT_FACES or ec not in STRAIGHT_FACES:
                return False
            length = ec - sc + 1
            if not (MIN_STRAIGHT_LEN <= length <= MAX_STRAIGHT_LEN):
                return False
            return not tc

        # 炸弹（纯四张，打出后立即加分，可压制所有非炸弹）
        elif f == FORMATIONS['bomb']:
            return sc in QUAD_FACES and ec is None and not tc

        return False

    def assert_legal(self):
        '''断言动作合法'''

        if not self.is_legal:
            raise RuntimeError(
                f'非法动作：{FORMATION_NAMES.get(self.formation, self.formation)}, '
                f'start={self.start_card}, end={self.end_card}, tail={self.tail_cards}')

    def deserialise(self) -> list[int]:
        '''
        将动作反序列化为牌值列表

        :return: 升序牌值列表
        '''

        self.assert_legal()
        f = self.formation

        if f == FORMATIONS['none']:
            return []

        multiple = {
            FORMATIONS['single']:   1,
            FORMATIONS['pocket']:   2,
            FORMATIONS['s_pocket']: 2,
            FORMATIONS['trips']:    3,
            FORMATIONS['s_trips']:  3,
            FORMATIONS['quads']:    4,
            FORMATIONS['straight']: 1,
            FORMATIONS['bomb']:     4,
        }[f]

        end = self.end_card if self.end_card is not None else self.start_card
        cards = [face for face in range(self.start_card, end + 1) for _ in range(multiple)]

        if self.tail_cards:
            cards += list(self.tail_cards)

        return sorted(cards)

    def __gt__(self, another: 'Action') -> bool:
        '''
        判断该动作是否大于另一动作

        比较规则：
        - 炸弹大于所有非炸弹；更大的炸弹（牌值更高）大于更小的炸弹
        - 其他情况：牌型须相同（formation 与 num_kicker 均须一致）；
          顺牌型还须长度相同；最终比较 start_card
        '''

        self.assert_legal()
        another.assert_legal()

        if self.formation == FORMATIONS['none'] or another.formation == FORMATIONS['none']:
            raise RuntimeError('不出牌的动作不参与比较')

        # 炸弹比较
        if self.formation == FORMATIONS['bomb'] and another.formation == FORMATIONS['bomb']:
            return self.start_card > another.start_card
        if self.formation == FORMATIONS['bomb']:
            return True
        if another.formation == FORMATIONS['bomb']:
            return False

        # 牌型须相同
        if self.formation != another.formation:
            raise RuntimeError(
                f'牌型不符：{FORMATION_NAMES.get(self.formation)} vs '
                f'{FORMATION_NAMES.get(another.formation)}')

        # 带牌数量须相同
        if self.num_kicker != another.num_kicker:
            raise RuntimeError(
                f'带牌数量不符：{self.num_kicker} vs {another.num_kicker}')

        # 顺牌型须长度相同
        if self.formation in {FORMATIONS['s_pocket'], FORMATIONS['s_trips'], FORMATIONS['straight']}:
            if self.length != another.length:
                raise RuntimeError(
                    f'顺牌型长度不符：{self.length} vs {another.length}')

        return self.start_card > another.start_card

    def __str__(self) -> str:
        cards = self.deserialise()
        return cards_to_str(cards)
