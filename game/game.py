import random

import sys
import os
path_root = os.path.dirname(os.path.dirname(__file__))
if path_root not in sys.path:
    sys.path.append(path_root)
path_current = os.path.dirname(__file__)
if path_current in sys.path:
    sys.path.remove(path_current)

from game import config as GAME
from game.config import FORMATIONS, HAND_SIZE
from game.action import Action
from game.player import Player


class Game(object):
    '''
    两人跑得快游戏牌局
    '''

    def __init__(self):
        self.players: list[Player] = None
        self.actions: list[Action] = None
        self.remain_cards: dict[int, int] = None
        self._current_player_index: int = None
        self._bomb_bonuses: list[int] = None
        self.num_bomb: int = None
        self.is_end: bool = None
        self._is_spring: bool = None

    # ── 属性 ────────────────────────────────────────────────────────────────

    @property
    def current_player_index(self) -> int:
        '''当前须做出动作的玩家序号'''
        return self._current_player_index

    @property
    def previous_action(self) -> Action:
        '''
        前序有效动作（当前玩家须超越的动作）。

        两人游戏中，只需看最近一个动作：
        - 若最近动作为不出（NONE），则为新一轮，返回 None
        - 否则返回最近的出牌动作
        '''
        if not self.actions:
            return None
        last = self.actions[-1]
        if last.formation == FORMATIONS['none']:
            return None
        return last

    @property
    def is_new_round(self) -> bool:
        '''当前是否为新一轮出牌'''
        return self.previous_action is None

    @property
    def is_spring(self) -> bool:
        '''
        游戏是否以春天结束（赢家在对手尚未打出任何牌时获胜）
        '''
        return self._is_spring if self.is_end else False

    # ── 方法 ────────────────────────────────────────────────────────────────

    def start(self, players: list[Player], first_player: int = None):
        '''
        开始新一局游戏

        :param players: 两位玩家列表，索引即玩家序号
        :param first_player: 先手玩家序号（0 或 1），None 时随机决定
        '''

        assert len(players) == 2
        self.players = players
        self.actions = []
        self.remain_cards = dict(GAME.DECK_COUNTS)
        self._bomb_bonuses = [0, 0]
        self.num_bomb = 0
        self.is_end = False
        self._current_player_index = first_player if first_player is not None else random.randint(0, 1)
        self._is_spring = False

    def act(self, action: Action):
        '''
        执行游戏动作

        :param action: 即将执行的动作
        '''

        if self.is_end:
            raise RuntimeError('游戏已结束')

        current_player = self.players[self._current_player_index]

        # 合法性检查
        if self.is_new_round:
            if action.formation == FORMATIONS['none']:
                raise RuntimeError('新一轮出牌时不可以不出牌')
        else:
            if action.formation != FORMATIONS['none']:
                if not action > self.previous_action:
                    raise RuntimeError(
                        f'出牌 {action} 未超越上家 {self.previous_action}')

        # 执行出牌
        current_player.act(action=action)
        self.actions.append(action)

        # 更新剩余牌
        for card in action.deserialise():
            self.remain_cards[card] -= 1

        # 炸弹即时分（zero-sum）：出炸弹方 +BOMB_BONUS，对方 -BOMB_BONUS
        if action.formation == FORMATIONS['bomb']:
            opp = 1 - self._current_player_index
            self._bomb_bonuses[self._current_player_index] += GAME.BOMB_BONUS
            self._bomb_bonuses[opp] -= GAME.BOMB_BONUS
            self.num_bomb += 1

        # 判断游戏是否结束
        if current_player.is_done:
            self._end_game(winner=self._current_player_index)
            return

        # 切换玩家（无论出牌还是不出牌，均切换）
        self._current_player_index = 1 - self._current_player_index

    def _end_game(self, winner: int):
        '''结束游戏并计算得分'''

        loser = 1 - winner

        # 春天：败者从未出过牌
        loser_player = self.players[loser]
        self._is_spring = all(count == 0 for count in loser_player.cards_out.values())

        # 基础分：春天时直接取春天分；否则按败者剩余手牌数计
        loser_remaining = len(loser_player.cards)
        if self._is_spring:
            base = GAME.SPRING_SCORE
        else:
            base = loser_remaining * GAME.CARD_SCORE

        # 炸弹分归打出炸弹的一方（与胜负无关）
        self.players[winner].is_winner = True
        self.players[winner].score = base + self._bomb_bonuses[winner]
        self.players[loser].is_winner = False
        self.players[loser].score = -base + self._bomb_bonuses[loser]

        self.is_end = True
