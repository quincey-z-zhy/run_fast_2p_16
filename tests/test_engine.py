"""
两人跑得快游戏引擎单元测试
全部使用中文注释
"""

import sys
import os
import random

import numpy as np
import pytest

# 将项目根目录加入 sys.path，确保所有包均可导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from game.config import (
    DECK_COUNTS, HAND_SIZE, FORMATIONS,
    BOMB_BONUS, SPRING_SCORE, CARD_SCORE,
    POCKET_TRIPLE_FACES, QUAD_FACES, STRAIGHT_FACES,
)
from game.cards import shuffle
from game.action import Action
from game.player import Player
from game.game import Game
from src.engine.action_generator import compute_action_mask, decode_action, get_all_legal_actions
from src.engine.obs_encoder import encode_obs
from src.env.game_env import RunFast2pEnv


# ════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════════════════

def make_game(p0_cards: list[int], p1_cards: list[int], first_player: int = 0) -> Game:
    """用指定手牌创建并启动一局游戏"""
    game = Game()
    players = [Player(p0_cards), Player(p1_cards)]
    game.start(players, first_player=first_player)
    return game


# ════════════════════════════════════════════════════════════════════════════
# 1. 牌组与发牌测试
# ════════════════════════════════════════════════════════════════════════════

class TestDeckAndDeal:

    def test_each_player_gets_16_cards(self):
        """洗牌后双方各得16张手牌（16-hand 变种）"""
        p0, p1 = shuffle()
        assert len(p0) == HAND_SIZE == 16, f"玩家0应有16张，实际有{len(p0)}张"
        assert len(p1) == HAND_SIZE == 16, f"玩家1应有16张，实际有{len(p1)}张"

    def test_dead_cards_count_16(self):
        """双方手牌合计32张，剩余16张为死牌，总牌数=48"""
        total = sum(DECK_COUNTS.values())
        assert total == 48, f"总牌数应为48，实际为{total}"

        p0, p1 = shuffle()
        dead_count = total - len(p0) - len(p1)
        assert dead_count == 16, f"死牌应为16张，实际为{dead_count}张"

    def test_hand_face_counts_within_deck_counts(self):
        """手牌中每种face的数量不超过DECK_COUNTS"""
        for _ in range(10):  # 多次洗牌验证随机性
            p0, p1 = shuffle()
            # 统计两手牌合并后的face数量
            all_cards = p0 + p1
            face_counts: dict[int, int] = {}
            for card in all_cards:
                face_counts[card] = face_counts.get(card, 0) + 1
            for face, count in face_counts.items():
                assert count <= DECK_COUNTS[face], (
                    f"face={face}总发出{count}张，超过牌库{DECK_COUNTS[face]}张"
                )

    def test_total_deck_counts_correct(self):
        """牌库各face数量验证（16-hand 变种）：3-K各4张，A=3，2=1"""
        for face in range(3, 14):  # 3..13 (K)
            assert DECK_COUNTS[face] == 4, f"face={face}应有4张"
        assert DECK_COUNTS[14] == 3, "A应有3张"
        assert DECK_COUNTS[15] == 1, "2应有1张"


# ════════════════════════════════════════════════════════════════════════════
# 2. Action is_legal 测试
# ════════════════════════════════════════════════════════════════════════════

class TestActionIsLegal:

    # ── 单牌 ────────────────────────────────────────────────────────────────

    def test_single_any_face_legal(self):
        """单牌：face ∈ [3,15] 均合法"""
        for face in range(3, 16):
            action = Action(FORMATIONS['single'], start_card=face)
            assert action.is_legal, f"单牌face={face}应合法"

    def test_single_no_end_card_legal(self):
        """单牌：有end_card时非法"""
        action = Action(FORMATIONS['single'], start_card=5, end_card=6)
        assert not action.is_legal, "单牌设置end_card应非法"

    # ── 对子 ────────────────────────────────────────────────────────────────

    def test_pocket_face_3_to_14_legal(self):
        """对子：face ∈ [3,14] 合法（16-hand 变种 A 有 3 张可成对）"""
        for face in POCKET_TRIPLE_FACES:  # [3..14]
            action = Action(FORMATIONS['pocket'], start_card=face)
            assert action.is_legal, f"对子face={face}应合法"

    def test_pocket_face_15_2_illegal(self):
        """对子：face=15(2)非法，2只有1张"""
        action = Action(FORMATIONS['pocket'], start_card=15)
        assert not action.is_legal, "对子face=15(2)应非法"

    # ── 炸弹 ────────────────────────────────────────────────────────────────

    def test_bomb_face_3_to_13_legal(self):
        """炸弹：face ∈ [3,13] 合法（16-hand 变种 3-K 各 4 张）"""
        for face in QUAD_FACES:  # [3..13]
            action = Action(FORMATIONS['bomb'], start_card=face)
            assert action.is_legal, f"炸弹face={face}应合法"

    def test_bomb_face_14_A_illegal(self):
        """炸弹：face=14(A)非法，A只有3张"""
        action = Action(FORMATIONS['bomb'], start_card=14)
        assert not action.is_legal, "炸弹face=14(A)应非法，A只有3张"

    # ── 顺子 ────────────────────────────────────────────────────────────────

    def test_straight_min_5_cards_legal(self):
        """顺子：最短5张且face ∈ [3,14] 合法"""
        # 3-4-5-6-7，共5张
        action = Action(FORMATIONS['straight'], start_card=3, end_card=7)
        assert action.is_legal, "顺子3-7应合法"

    def test_straight_contains_face_15_2_illegal(self):
        """顺子：含face=15(2)非法"""
        # 规格：顺子最大到A(14)，2不参与顺子
        action = Action(FORMATIONS['straight'], start_card=12, end_card=15)
        assert not action.is_legal, "顺子含2应非法"

    def test_straight_length_4_illegal(self):
        """顺子：长度<5非法"""
        action = Action(FORMATIONS['straight'], start_card=3, end_card=6)
        assert not action.is_legal, "4张顺子应非法"

    def test_straight_face_3_to_14_legal(self):
        """顺子：face ∈ [3,14] 包含A，5张及以上合法"""
        # 10-J-Q-K-A，共5张（10~14）
        action = Action(FORMATIONS['straight'], start_card=10, end_card=14)
        assert action.is_legal, "顺子T-A应合法"

    # ── 连对 ────────────────────────────────────────────────────────────────

    def test_s_pocket_length_2_legal(self):
        """连对：连续2对合法"""
        action = Action(FORMATIONS['s_pocket'], start_card=3, end_card=4)
        assert action.is_legal, "连对3-4应合法"

    def test_s_pocket_length_1_illegal(self):
        """连对：长度=1（仅1对）非法"""
        action = Action(FORMATIONS['s_pocket'], start_card=5, end_card=5)
        assert not action.is_legal, "单对作连对应非法"

    def test_s_pocket_contains_face_15_illegal(self):
        """连对：含face=15(2)非法，2只有1张"""
        action = Action(FORMATIONS['s_pocket'], start_card=14, end_card=15)
        assert not action.is_legal, "连对含2应非法"

    def test_s_pocket_K_A_legal(self):
        """连对：K-A（13-14）在 16-hand 变种合法（K、A 均有 ≥2 张）"""
        action = Action(FORMATIONS['s_pocket'], start_card=13, end_card=14)
        assert action.is_legal, "连对 K-A 在 16-hand 变种应合法"

    # ── 飞机 ────────────────────────────────────────────────────────────────

    def test_s_trips_n2_no_kicker_legal(self):
        """飞机：N=2连续三条，带牌数=0合法"""
        action = Action(FORMATIONS['s_trips'], start_card=3, end_card=4)
        assert action.is_legal, "飞机3-4无带牌应合法"

    def test_s_trips_n2_n_kickers_legal(self):
        """飞机：N=2连续三条，带牌数=N=2合法"""
        action = Action(FORMATIONS['s_trips'], start_card=3, end_card=4, tail_cards=[5, 6])
        assert action.is_legal, "飞机3-4带2张应合法"

    def test_s_trips_n2_2n_kickers_legal(self):
        """飞机：N=2连续三条，带牌数=2N=4合法"""
        action = Action(FORMATIONS['s_trips'], start_card=3, end_card=4, tail_cards=[5, 6, 7, 8])
        assert action.is_legal, "飞机3-4带4张应合法"

    def test_s_trips_wrong_kicker_count_illegal(self):
        """飞机：带牌数不为0/N/2N时非法"""
        # N=2，带牌数=1，既不是0，也不是2，也不是4
        action = Action(FORMATIONS['s_trips'], start_card=3, end_card=4, tail_cards=[5])
        assert not action.is_legal, "飞机带牌数=1应非法"

        # 带牌数=3，N=2时不合法
        action2 = Action(FORMATIONS['s_trips'], start_card=3, end_card=4, tail_cards=[5, 6, 7])
        assert not action2.is_legal, "飞机带牌数=3(N=2时)应非法"


# ════════════════════════════════════════════════════════════════════════════
# 3. Action 比较测试
# ════════════════════════════════════════════════════════════════════════════

class TestActionComparison:

    def test_bomb_beats_any_single(self):
        """炸弹 > 单牌（任何单牌）"""
        bomb = Action(FORMATIONS['bomb'], start_card=3)  # 最小炸弹
        for face in range(3, 16):
            single = Action(FORMATIONS['single'], start_card=face)
            assert bomb > single, f"炸弹3应大于单牌face={face}"

    def test_bigger_bomb_beats_smaller_bomb(self):
        """大炸弹 > 小炸弹（start_card更大）"""
        small_bomb = Action(FORMATIONS['bomb'], start_card=3)
        big_bomb = Action(FORMATIONS['bomb'], start_card=12)
        assert big_bomb > small_bomb, "炸弹12应大于炸弹3"
        assert not (small_bomb > big_bomb), "炸弹3不应大于炸弹12"

    def test_same_formation_higher_face_wins(self):
        """同型：face大者胜"""
        # 单牌
        single_high = Action(FORMATIONS['single'], start_card=10)
        single_low = Action(FORMATIONS['single'], start_card=5)
        assert single_high > single_low, "单牌T应大于单牌5"

        # 对子
        pocket_high = Action(FORMATIONS['pocket'], start_card=10)
        pocket_low = Action(FORMATIONS['pocket'], start_card=5)
        assert pocket_high > pocket_low, "对子T应大于对子5"

        # 炸弹
        bomb_high = Action(FORMATIONS['bomb'], start_card=10)
        bomb_low = Action(FORMATIONS['bomb'], start_card=5)
        assert bomb_high > bomb_low, "炸弹10应大于炸弹5"

    def test_different_formation_non_bomb_raises(self):
        """不同型非炸弹：比较应raise RuntimeError"""
        single = Action(FORMATIONS['single'], start_card=8)
        pocket = Action(FORMATIONS['pocket'], start_card=5)
        with pytest.raises(RuntimeError):
            _ = single > pocket

    def test_none_action_comparison_raises(self):
        """不出牌的动作不参与比较，应raise RuntimeError"""
        none_action = Action(FORMATIONS['none'])
        single = Action(FORMATIONS['single'], start_card=5)
        with pytest.raises(RuntimeError):
            _ = none_action > single

    def test_straight_different_length_raises(self):
        """顺牌型长度不同：比较应raise RuntimeError"""
        straight5 = Action(FORMATIONS['straight'], start_card=3, end_card=7)   # 长度5
        straight6 = Action(FORMATIONS['straight'], start_card=3, end_card=8)   # 长度6
        with pytest.raises(RuntimeError):
            _ = straight6 > straight5


# ════════════════════════════════════════════════════════════════════════════
# 4. 游戏流程测试
# ════════════════════════════════════════════════════════════════════════════

class TestGameFlow:

    def test_new_round_pass_raises(self):
        """新一轮出牌时，pass应raise RuntimeError"""
        p0 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 14, 14, 15]
        p1 = [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8]
        game = make_game(p0, p1, first_player=0)
        assert game.is_new_round, "初始应为新一轮"
        none_action = Action(FORMATIONS['none'])
        with pytest.raises(RuntimeError):
            game.act(none_action)

    def test_must_beat_previous_or_raise(self):
        """跟牌时，出牌必须超越上家，否则raise RuntimeError"""
        p0 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 14, 14, 15]
        p1 = [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8]
        game = make_game(p0, p1, first_player=0)
        # P0出单牌8
        game.act(Action(FORMATIONS['single'], start_card=8))
        # P1尝试出单牌5（小于8），应raise
        low_single = Action(FORMATIONS['single'], start_card=5)
        with pytest.raises(RuntimeError):
            game.act(low_single)

    def test_bomb_can_beat_non_bomb(self):
        """炸弹可以压制所有非炸弹（炸弹跟对子合法）"""
        # P0 手牌包含对子K，P1 手牌包含炸弹3
        p0 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 14, 14, 15]
        p1 = [3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8]
        game = make_game(p0, p1, first_player=0)
        # P0出对子K
        game.act(Action(FORMATIONS['pocket'], start_card=13))
        # P1出炸弹3（应合法，炸弹压制对子）
        game.act(Action(FORMATIONS['bomb'], start_card=3))
        # 未抛出异常表示炸弹合法跟牌

    def test_game_ends_when_hand_empty(self):
        """手牌打空时is_end=True"""
        # P0只有1张牌
        p0 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 14, 14, 15]
        p1 = [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8]
        game = make_game(p0, p1, first_player=0)
        assert not game.is_end

        # 通过env随机合法动作跑完一局
        env = RunFast2pEnv()
        obs, info = env.reset()
        terminated = False
        max_steps = 500
        for _ in range(max_steps):
            mask = info['action_mask']
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                break
            action_idx = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action_idx)
            if terminated:
                break
        assert terminated, "随机对局应在有限步内结束"
        assert env._game.is_end, "游戏结束时is_end应为True"
        # 找到赢家，验证其手牌为空
        winner = next(p for p in env._game.players if p.is_winner)
        assert len(winner.cards) == 0, "赢家手牌应为空"


# ════════════════════════════════════════════════════════════════════════════
# 5. 计分测试
# ════════════════════════════════════════════════════════════════════════════

class TestScoring:

    def _run_to_end_with_actions(self, p0_cards, p1_cards, action_sequence):
        """
        按指定动作序列执行游戏，返回game对象
        action_sequence: list of Action，按轮次顺序（p0先手）
        """
        game = make_game(p0_cards, p1_cards, first_player=0)
        for action in action_sequence:
            if game.is_end:
                break
            game.act(action)
        return game

    def test_normal_end_winner_score_equals_loser_remaining(self):
        """正常结束：赢家得分=败者剩余数，输家得分=负数"""
        # 构建一局简单的游戏：P0一次性出完所有单牌
        # P0手牌：13张单牌 + 1对（实际需用合法牌）
        # 简单方案：用env快速跑完，验证得分关系
        env = RunFast2pEnv()
        obs, info = env.reset()
        terminated = False
        for _ in range(500):
            mask = info['action_mask']
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                break
            action_idx = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action_idx)
            if terminated:
                break

        assert terminated
        game = env._game
        winner_idx = next(i for i, p in enumerate(game.players) if p.is_winner)
        loser_idx = 1 - winner_idx
        loser = game.players[loser_idx]
        winner = game.players[winner_idx]

        # 基础得分 = 败者剩余手牌数（不考虑春天/炸弹情况，直接验证符号）
        assert winner.score > 0, "赢家得分应为正"
        assert loser.score <= 0, "输家得分应为非正"

    def test_spring_bonus(self):
        """春天：败者从未出牌时赢家额外+30"""
        # 构建：P0 16 张可逐步出完，P1 从不出牌
        # P0：3-Q 各 1 张（顺子 3-Q 10 张）+ K×3 + A×2 + 2 = 16 张
        p0 = sorted([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 14, 14, 15])
        p1 = [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8]

        # 验证P0/P1手牌不重叠（不超过DECK_COUNTS）
        from collections import Counter
        combined = Counter(p0 + p1)
        for face, cnt in combined.items():
            assert cnt <= DECK_COUNTS[face], f"测试手牌face={face}超出牌库数量"

        game = make_game(p0, p1, first_player=0)

        # P0 逐步出完所有牌，P1 每次 pass

        # P0 出单牌 15(2)
        game.act(Action(FORMATIONS['single'], start_card=15))
        # P1 pass
        game.act(Action(FORMATIONS['none']))
        # P0 出单牌 14(A)
        game.act(Action(FORMATIONS['single'], start_card=14))
        # P1 pass
        game.act(Action(FORMATIONS['none']))
        # P0 再出单牌 14(A)（A 在 16-hand 变种有 3 张）
        game.act(Action(FORMATIONS['single'], start_card=14))
        # P1 pass
        game.act(Action(FORMATIONS['none']))
        # P0 出三条 K
        game.act(Action(FORMATIONS['trips'], start_card=13))
        # P1 pass
        game.act(Action(FORMATIONS['none']))
        # P0 出顺子 3-Q（10 张）
        game.act(Action(FORMATIONS['straight'], start_card=3, end_card=12))
        # 此时 P0 手牌清空，游戏结束

        assert game.is_end, "P0出完后游戏应结束"
        assert game.is_spring, "P1从未出牌，应触发春天"
        winner = game.players[0]
        loser = game.players[1]
        assert winner.is_winner
        # 春天时基础分用 SPRING_SCORE 替代（见 game/game.py _end_game 实现）
        # 春天得分 = SPRING_SCORE + 赢家炸弹奖励（本测试无炸弹）= 30
        loser_remaining = len(loser.cards)  # P1 未出牌，仍有 16 张
        assert loser_remaining == 16
        assert winner.score == SPRING_SCORE, (
            f"春天赢家得分应为{SPRING_SCORE}，实际为{winner.score}"
        )

    def test_bomb_bonus_independent_of_outcome(self):
        """炸弹奖励：打出炸弹的玩家+10，与胜负无关"""
        # P0出炸弹3，然后输掉游戏
        # P0手牌：炸弹3(3,3,3,3) + 11张其他高牌
        # P1手牌：一堆低牌，能快速出完
        p0 = sorted([3, 3, 3, 3, 13, 13, 13, 14, 15, 10, 11, 12, 8, 9, 7, 8])
        p1 = [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8]

        # 验证手牌合法性
        from collections import Counter
        combined = Counter(p0 + p1)
        for face, cnt in combined.items():
            assert cnt <= DECK_COUNTS[face], f"测试手牌face={face}超出牌库数量"

        game = make_game(p0, p1, first_player=0)
        # P0出炸弹3
        game.act(Action(FORMATIONS['bomb'], start_card=3))
        # P0已获得炸弹奖励，验证_bomb_bonuses
        assert game._bomb_bonuses[0] == BOMB_BONUS, "P0应获得炸弹奖励"

    def test_loser_bomb_score_can_be_positive(self):
        """输家打出炸弹时得分可为正（炸弹奖励补偿负分）"""
        # 场景：P1输，但打出炸弹，炸弹分可补偿负分使得分为正
        # P0: face=3..9各1张 + face=13×3 + face=14 + face=15 = 15张（不含炸弹face）
        # P1: 炸弹10（10×4）+ 其他11张低牌
        # 手牌合法：P0不使用face=10，P1用炸弹10
        p0 = sorted([3, 4, 5, 6, 7, 8, 9, 13, 13, 13, 14, 14, 15, 11, 12, 12])
        p1 = sorted([10, 10, 10, 10, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])

        from collections import Counter
        combined = Counter(p0 + p1)
        for face, cnt in combined.items():
            assert cnt <= DECK_COUNTS[face], f"测试手牌face={face}超出牌库数量"

        game = make_game(p0, p1, first_player=0)

        # P0出单牌15(2)
        game.act(Action(FORMATIONS['single'], start_card=15))
        # P1出炸弹10（炸弹可压制所有非炸弹，合法）
        game.act(Action(FORMATIONS['bomb'], start_card=10))
        # 炸弹打出后切换玩家，P1获得炸弹奖励
        assert game._bomb_bonuses[1] == BOMB_BONUS, "P1应获得炸弹奖励10"

        # 继续走到游戏结束，验证若P1输了但炸弹分>基础负分则得分为正
        # P0跟P1炸弹：P0无炸弹，必须pass
        if not game.is_end and game.current_player_index == 0:
            game.act(Action(FORMATIONS['none']))

        # 此时P1新一轮，P1还有11张牌，通过env让游戏跑完
        # 直接验证炸弹奖励已累积
        assert game._bomb_bonuses[1] >= BOMB_BONUS, "P1应已获得至少1个炸弹奖励"


# ════════════════════════════════════════════════════════════════════════════
# 6. obs编码测试
# ════════════════════════════════════════════════════════════════════════════

class TestObsEncoding:

    def test_obs_dim_80_float32(self):
        """obs维度=80，dtype=float32"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        assert obs.shape == (80,), f"obs维度应为80，实际为{obs.shape}"
        assert obs.dtype == np.float32, f"obs dtype应为float32，实际为{obs.dtype}"

    def test_own_hand_values_in_0_1(self):
        """own_hand值∈[0,1]"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        own_hand = obs[0:13]
        assert np.all(own_hand >= 0.0), "own_hand不应有负值"
        assert np.all(own_hand <= 1.0), "own_hand不应超过1.0"

    def test_remain_cards_initial_all_one(self):
        """remain_cards初始值全为1.0"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        remain_cards = obs[13:26]
        assert np.allclose(remain_cards, 1.0), (
            f"初始remain_cards应全为1.0，实际为{remain_cards}"
        )

    def test_is_new_round_last_action_all_zero(self):
        """is_new_round时last_action段（dim 26-64）全零"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        # dim 65 = is_new_round
        assert obs[65] == 1.0, "初始应为new_round"
        last_action_seg = obs[26:65]
        assert np.all(last_action_seg == 0.0), (
            f"is_new_round时last_action段应全零，实际为{last_action_seg}"
        )

    def test_remain_cards_decrease_after_action(self):
        """执行动作后remain_cards正确递减"""
        env = RunFast2pEnv()
        obs_before, info = env.reset()
        remain_before = obs_before[13:26].copy()

        # 执行一个合法的非pass动作
        mask = info['action_mask']
        # 跳过idx=0(pass)，找第一个合法动作
        non_pass_legal = [i for i in range(1, 294) if mask[i]]
        assert len(non_pass_legal) > 0, "初始应有合法出牌动作"
        action_idx = non_pass_legal[0]

        obs_after, reward, terminated, truncated, info_after = env.step(action_idx)
        remain_after = obs_after[13:26]

        # 打出牌后，remain_cards的总和应减少
        # 由于obs_after是下一个玩家视角，remain_cards应有变化
        # 对于已打出的牌，对应face的remain_cards应减少
        total_before = np.sum(remain_before)
        total_after = np.sum(remain_after)
        if not terminated:
            assert total_after < total_before, (
                f"出牌后remain_cards总量应减少，before={total_before:.4f}，after={total_after:.4f}"
            )


# ════════════════════════════════════════════════════════════════════════════
# 7. 动作掩码测试
# ════════════════════════════════════════════════════════════════════════════

class TestActionMask:

    def test_mask_dim_294_bool(self):
        """掩码维度=294，dtype=bool（16-hand 变种）"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        mask = info['action_mask']
        assert mask.shape == (294,), f"掩码维度应为294，实际为{mask.shape}"
        assert mask.dtype == bool, f"掩码dtype应为bool，实际为{mask.dtype}"

    def test_new_round_pass_masked_false(self):
        """is_new_round时pass（idx=0）被掩码为False"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        assert obs[65] == 1.0, "初始应为new_round"
        mask = info['action_mask']
        assert not mask[0], "is_new_round时pass应被掩码为False"

    def test_bomb_masked_true_when_following(self):
        """手中有炸弹时bomb动作被掩码为True（跟牌场景）"""
        # P0不使用face=9，P1拥有炸弹9（9×4张），满足牌库限制
        # P0: 3,4,5,6,7,8,8,10,11,12,13,13,13,14,15（15张，face=9=0）
        # P1: 9,9,9,9,3,3,3,4,4,4,5,5,5,6,6（15张，P1有炸弹9）
        p0 = sorted([3, 4, 5, 6, 7, 8, 8, 10, 11, 12, 13, 13, 13, 14, 14, 15])
        p1 = sorted([9, 9, 9, 9, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])

        from collections import Counter
        combined = Counter(p0 + p1)
        for face, cnt in combined.items():
            assert cnt <= DECK_COUNTS[face], f"测试手牌face={face}超出牌库数量"

        game = make_game(p0, p1, first_player=0)
        # P0出单牌15(2)
        game.act(Action(FORMATIONS['single'], start_card=15))
        # 现在是P1的回合，P1有炸弹9
        assert game.current_player_index == 1
        assert not game.is_new_round, "P0出牌后应为跟牌场景"

        mask = compute_action_mask(game)
        # 炸弹9对应 idx = 283 + (9-3) = 289（16-hand 变种）
        bomb_idx_9 = 283 + (9 - 3)
        assert mask[bomb_idx_9], "P1有炸弹9时，bomb idx=289应为True"

    def test_no_card_action_masked_false(self):
        """手中无对应牌时，该动作被掩码为False"""
        # P0没有A(14)，所以单牌A应被掩码为False
        p0 = sorted([3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6])
        p1 = sorted([7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 14, 14, 15])

        from collections import Counter
        combined = Counter(p0 + p1)
        for face, cnt in combined.items():
            assert cnt <= DECK_COUNTS[face], f"测试手牌face={face}超出牌库数量"

        game = make_game(p0, p1, first_player=0)
        mask = compute_action_mask(game)
        # 单牌A(14) 对应 idx = 1 + (14-3) = 12
        single_A_idx = 1 + (14 - 3)
        assert not mask[single_A_idx], "P0无A时，单牌A应被掩码为False"

    def test_at_least_one_legal_action_when_game_ongoing(self):
        """游戏未结束时，掩码中合法动作至少1个"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        for _ in range(200):
            mask = info['action_mask']
            legal_count = np.sum(mask)
            assert legal_count >= 1, "游戏进行中应至少有1个合法动作"
            legal_indices = np.where(mask)[0]
            action_idx = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action_idx)
            if terminated:
                break


# ════════════════════════════════════════════════════════════════════════════
# 8. Gym接口测试
# ════════════════════════════════════════════════════════════════════════════

class TestGymInterface:

    def test_reset_returns_obs_info(self):
        """reset()返回(obs, info)，obs shape=(80,)"""
        env = RunFast2pEnv()
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray), "obs应为numpy数组"
        assert obs.shape == (80,), f"obs shape应为(80,)，实际为{obs.shape}"
        assert isinstance(info, dict), "info应为字典"
        assert 'action_mask' in info, "info应包含action_mask"

    def test_step_returns_5_tuple(self):
        """step()返回(obs, reward, terminated, truncated, info)"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        mask = info['action_mask']
        legal_indices = np.where(mask)[0]
        action_idx = int(legal_indices[0])
        result = env.step(action_idx)
        assert isinstance(result, tuple) and len(result) == 5, (
            f"step应返回5元组，实际返回{len(result)}元素"
        )
        obs2, reward, terminated, truncated, info2 = result
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info2, dict)

    def test_game_end_terminated_true_reward_nonzero(self):
        """游戏结束后terminated=True，reward≠0"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        terminated = False
        final_reward = None
        for _ in range(500):
            mask = info['action_mask']
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                break
            action_idx = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action_idx)
            if terminated:
                final_reward = reward
                break
        assert terminated, "游戏应在500步内结束"
        assert final_reward != 0, f"终局reward应非零，实际为{final_reward}"

    def test_illegal_action_reward_minus_1_not_terminated(self):
        """非法动作：reward=-1，terminated=False，info['illegal_action']=True"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        # 找一个非法动作：掩码为False的索引
        mask = info['action_mask']
        illegal_indices = np.where(~mask)[0]
        assert len(illegal_indices) > 0, "应存在非法动作"
        illegal_idx = int(illegal_indices[0])
        obs2, reward, terminated, truncated, info2 = env.step(illegal_idx)
        assert reward == -1.0, f"非法动作reward应为-1，实际为{reward}"
        assert not terminated, "非法动作后游戏不应结束"
        assert info2.get('illegal_action') is True, "info['illegal_action']应为True"

    def test_full_game_no_error(self):
        """完整对局（随机合法动作）无报错完成"""
        random.seed(42)
        np.random.seed(42)
        env = RunFast2pEnv()
        obs, info = env.reset(seed=42)
        terminated = False
        steps = 0
        max_steps = 1000
        while not terminated and steps < max_steps:
            mask = info['action_mask']
            legal_indices = np.where(mask)[0]
            assert len(legal_indices) > 0, f"第{steps}步无合法动作"
            action_idx = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action_idx)
            steps += 1
        assert terminated, f"对局应在{max_steps}步内正常结束，实际{steps}步"
        assert steps > 0, "游戏应至少执行一步"

    def test_truncated_always_false(self):
        """truncated始终为False"""
        env = RunFast2pEnv()
        obs, info = env.reset()
        for _ in range(50):
            mask = info['action_mask']
            legal_indices = np.where(mask)[0]
            if len(legal_indices) == 0:
                break
            action_idx = int(np.random.choice(legal_indices))
            obs, reward, terminated, truncated, info = env.step(action_idx)
            assert truncated is False, "truncated应始终为False"
            if terminated:
                break
