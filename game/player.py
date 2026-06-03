import sys
import os
path_root = os.path.dirname(os.path.dirname(__file__))
if path_root not in sys.path:
    sys.path.append(path_root)
path_current = os.path.dirname(__file__)
if path_current in sys.path:
    sys.path.remove(path_current)

from game.config import FACES
from game.action import Action


class Player(object):
    '''
    游戏玩家
    '''

    def __init__(self, cards: list[int]):
        '''
        游戏玩家，初始化

        :param cards: 初始手牌
        '''

        self.cards = sorted(cards)

        # 各牌值已出牌数量
        self.cards_out = {face: 0 for face in FACES}

        # 是否为当前局的赢家
        self.is_winner: bool = None

        # 游戏得分（正数为赢，负数为输）
        self.score: int = 0

    @property
    def is_done(self) -> bool:
        '''手牌是否已全部打出'''
        return not self.cards

    def act(self, action: Action):
        '''
        执行出牌动作，从手牌中移除对应的牌

        :param action: 出牌动作
        '''

        action.assert_legal()
        action_cards = action.deserialise()

        for card in action_cards:
            self.cards.remove(card)
            self.cards_out[card] += 1
